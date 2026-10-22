// leak_harness: reproduit le cycle ouverture/fermeture de preview Staxrip dans
// un seul processus. Chaque cycle : creation d'un core VapourSynth, chargement
// du plugin, filtrage de quelques images, destruction du core. La memoire
// privee du processus est mesuree apres chaque cycle : la fuite DRAM observee
// dans Staxrip apparait comme une croissance qui ne redescend pas.
// Usage: leak_harness W H bits fmt cycles [key=val ...]
//   fmt: GRAY | YUV420 | YUV422 | YUV444 | RGB
//   key=val: arguments du filtre (bt=4 sigma=3 ...); les images sources sont
//   generees (bruit deterministe), aucun fichier requis.
//   ref=<dll> refns=<ns> reffn=<fn> noplugin=1 : mesure fft3dcl a la place.

#include <windows.h>
#include <psapi.h>
#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static std::vector<unsigned char> g_raw; // concatenated packed planes, per frame
static int g_w = 0, g_h = 0, g_bpp = 1, g_ssw = 0, g_ssh = 0, g_nplanes = 1;
static size_t g_framebytes = 0;
static int g_frames = 4;
static VSVideoFormat g_fmt {};

static const VSFrame *VS_CC srcGetFrame(int n, int ar, void *, void **,
                                        VSFrameContext *, VSCore *core, const VSAPI *vsapi) {
  if (ar != arInitial && ar != arAllFramesReady) return nullptr;
  VSFrame *f = vsapi->newVideoFrame(&g_fmt, g_w, g_h, nullptr, core);
  size_t base = (size_t)n * g_framebytes;
  size_t off = base;
  for (int p = 0; p < g_nplanes; p++) {
    int pw = p == 0 ? g_w : g_w >> g_ssw;
    int ph = p == 0 ? g_h : g_h >> g_ssh;
    ptrdiff_t st = vsapi->getStride(f, p);
    uint8_t *d = vsapi->getWritePtr(f, p);
    for (int y = 0; y < ph; y++)
      memcpy(d + (size_t)y * st, g_raw.data() + off + (size_t)y * pw * g_bpp, (size_t)pw * g_bpp);
    off += (size_t)pw * ph * g_bpp;
  }
  return f;
}
static void VS_CC srcFree(void *, VSCore *, const VSAPI *) {}

static size_t priv_bytes() {
  PROCESS_MEMORY_COUNTERS_EX pmc {};
  pmc.cb = sizeof(pmc);
  GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc));
  return pmc.PrivateUsage;
}

static LONG WINAPI av_report(EXCEPTION_POINTERS *ep) {
  void *addr = ep->ExceptionRecord->ExceptionAddress;
  char name[MAX_PATH] = "?";
  HMODULE mod = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         (char *)addr, &mod) && mod)
    GetModuleFileNameA(mod, name, MAX_PATH);
  fprintf(stderr, "ACCESS VIOLATION at %p in %s (code %08lx, base %p)\n", addr, name,
          (unsigned long)ep->ExceptionRecord->ExceptionCode, (void *)mod);
  fflush(stderr);
  return EXCEPTION_EXECUTE_HANDLER;
}

int main(int argc, char **argv) {
  SetUnhandledExceptionFilter(av_report);
  if (argc < 6) {
    fprintf(stderr, "usage: leak_harness W H bits fmt cycles [key=val ...]\n");
    return 1;
  }
  int W = atoi(argv[1]), H = atoi(argv[2]), bits = atoi(argv[3]);
  std::string fmt = argv[4];
  int cycles = atoi(argv[5]);
  if (cycles < 1) cycles = 1;

  int cf = cfGray, ssw = 0, ssh = 0, npl = 1;
  if (fmt == "GRAY") {}
  else if (fmt == "YUV420") { cf = cfYUV; ssw = 1; ssh = 1; npl = 3; }
  else if (fmt == "YUV422") { cf = cfYUV; ssw = 1; ssh = 0; npl = 3; }
  else if (fmt == "YUV444") { cf = cfYUV; ssw = 0; ssh = 0; npl = 3; }
  else if (fmt == "RGB")    { cf = cfRGB; npl = 3; }
  else { fprintf(stderr, "unknown format %s\n", fmt.c_str()); return 1; }

  g_w = W; g_h = H; g_ssw = ssw; g_ssh = ssh; g_nplanes = npl;
  g_bpp = bits <= 8 ? 1 : (bits <= 16 ? 2 : 4);
  g_framebytes = 0;
  for (int p = 0; p < npl; p++) {
    int pw = p == 0 ? W : W >> ssw;
    int ph = p == 0 ? H : H >> ssh;
    g_framebytes += (size_t)pw * ph * g_bpp;
  }

  std::vector<std::pair<std::string, std::string>> kv;
  for (int i = 6; i < argc; i++) {
    std::string s = argv[i];
    size_t eq = s.find('=');
    if (eq != std::string::npos) kv.push_back({s.substr(0, eq), s.substr(eq + 1)});
  }
  std::string refdll, reffn = "FFT3DGPU", refns = "fft3dgpu";
  bool have_ref = false;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "ref") { refdll = kv[i].second; have_ref = true; kv.erase(kv.begin() + i); break; }
  bool nofilter = false, nofetch = false;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "nofilter") { nofilter = true; kv.erase(kv.begin() + i); break; }
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "nofetch") { nofetch = true; kv.erase(kv.begin() + i); break; }
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "frames") { g_frames = atoi(kv[i].second.c_str()); kv.erase(kv.begin() + i); break; }
  int fetch_override = -1;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "fetch") { fetch_override = atoi(kv[i].second.c_str()); kv.erase(kv.begin() + i); break; }
  int fetch_start = 0;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "fetchstart") { fetch_start = atoi(kv[i].second.c_str()); kv.erase(kv.begin() + i); break; }
  bool stdcopy = false;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "stdcopy") { stdcopy = true; kv.erase(kv.begin() + i); break; }
  bool onecore = false;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "onecore") { onecore = true; kv.erase(kv.begin() + i); break; }
  int nthreads = -1;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "threads") { nthreads = atoi(kv[i].second.c_str()); kv.erase(kv.begin() + i); break; }
  bool vmmap = false;
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "vmmap") { vmmap = true; kv.erase(kv.begin() + i); break; }
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "reffn") { reffn = kv[i].second; kv.erase(kv.begin() + i); break; }
  for (size_t i = 0; i < kv.size(); i++)
    if (kv[i].first == "refns") { refns = kv[i].second; kv.erase(kv.begin() + i); break; }

  // source synthetique : bruit deterministe (comme une image tres bruitee).
  // genere APRES le parsing de frames= pour dimensionner g_raw correctement.
  g_raw.assign((size_t)g_frames * g_framebytes, 0);
  {
    std::mt19937 rng(12345);
    for (size_t i = 0; i < g_raw.size(); i += 2) {
      uint16_t v = (uint16_t)(rng() & (bits <= 8 ? 0xFF : 0x3FF));
      memcpy(&g_raw[i], &v, 2);
    }
  }

  const char *VSDLL = "C:\\Users\\Bruno\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\vapoursynth\\libvapoursynth.dll";
  HMODULE vs = LoadLibraryA(VSDLL);
  if (!vs) { fprintf(stderr, "LoadLibrary VapourSynth failed: %lu\n", GetLastError()); return 1; }
  auto getAPI = (VSGetVapourSynthAPI)GetProcAddress(vs, "getVapourSynthAPI");
  if (!getAPI) { fprintf(stderr, "getVapourSynthAPI not found\n"); return 1; }
  const VSAPI *vsapi = getAPI(VS_MAKE_VERSION(4, 3));
  if (!vsapi) { fprintf(stderr, "API 4.3 not available\n"); return 1; }

  const char *ourdll = "C:\\Users\\Bruno\\Desktop\\OpenCode\\FFT3DNEOCL\\fft3dneocl\\build\\Release\\fft3dneocl.dll";

  printf("cycle  prive (Mo)   delta (Mo)\n");
  fflush(stdout);
  size_t prev = priv_bytes();
  VSCore *core = nullptr;
  VSPlugin *stdp = nullptr, *plug = nullptr;
  for (int c = 0; c < cycles; c++) {
    if (!onecore || c == 0) {
      core = vsapi->createCore(0);
      if (!core) { fprintf(stderr, "createCore failed\n"); return 1; }
      if (nthreads >= 0) vsapi->setThreadCount(nthreads, core);
      stdp = vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core);
      {
        VSMap *a = vsapi->createMap();
        vsapi->mapSetData(a, "path", have_ref ? refdll.c_str() : ourdll, -1, dtUtf8, maReplace);
        VSMap *o = vsapi->invoke(stdp, "LoadPlugin", a);
        if (vsapi->mapGetError(o)) { fprintf(stderr, "LoadPlugin: %s\n", vsapi->mapGetError(o)); return 1; }
        vsapi->freeMap(a); vsapi->freeMap(o);
      }
      plug = have_ref ? vsapi->getPluginByNamespace(refns.c_str(), core)
                      : vsapi->getPluginByID("com.github.fft3dneocl", core);
      if (!plug) { fprintf(stderr, "plugin not loaded\n"); return 1; }
    }
    size_t m_core = priv_bytes();
    size_t m_load = m_core;

    VSVideoInfo vi {};
    vi.width = W; vi.height = H; vi.numFrames = g_frames; vi.fpsNum = 24; vi.fpsDen = 1;
    if (!vsapi->queryVideoFormat(&vi.format, cf, bits == 32 ? stFloat : stInteger, bits, ssw, ssh, core)) {
      fprintf(stderr, "queryVideoFormat failed\n"); return 1;
    }
    g_fmt = vi.format;
    VSNode *src = vsapi->createVideoFilter2("RawSrc", &vi, srcGetFrame, srcFree, fmParallel, nullptr, 0, nullptr, core);
    if (!src) { fprintf(stderr, "source node failed\n"); return 1; }

    VSMap *a = vsapi->createMap();
    vsapi->mapSetNode(a, "clip", src, maReplace);
    for (auto &pr : kv) {
      if (have_ref) { // l'original fft3dcl ne connait pas ces arguments
        static const char *ref_skip[] = {"planes","device_id","platform_id","num_streams",
          "l","t","r","b","opt","ncpu","mt","measure","pframe","px","py","pshow",
          "pcutoff","pfactor","hr","ht","dehalo"};
        bool skip = false;
        for (const char *s : ref_skip) if (pr.first == s) { skip = true; break; }
        if (skip) continue;
      }
      static const char *float_names[] = {"sigma","sigma2","sigma3","sigma4","beta","sharpen",
        "scutoff","svr","smin","smax","kratio","degrid","pcutoff","pfactor","hr","ht","dehalo"};
      bool isfloat = false;
      for (const char *s : float_names) if (pr.first == s) { isfloat = true; break; }
      bool isint = !isfloat && !pr.second.empty();
      for (size_t j = (pr.second[0] == '-' ? 1 : 0); j < pr.second.size(); j++)
        if (!isdigit((unsigned char)pr.second[j])) isint = false;
      if (pr.first == "planes") {
        std::vector<int64_t> pv;
        size_t pos = 0;
        while (pos < pr.second.size()) {
          size_t cm = pr.second.find(',', pos);
          pv.push_back(atoll(pr.second.substr(pos, cm == std::string::npos ? std::string::npos : cm - pos).c_str()));
          if (cm == std::string::npos) break;
          pos = cm + 1;
        }
        vsapi->mapSetIntArray(a, pr.first.c_str(), pv.data(), (int)pv.size());
      } else if (isint) {
        vsapi->mapSetInt(a, pr.first.c_str(), atoll(pr.second.c_str()), maReplace);
      } else {
        vsapi->mapSetFloat(a, pr.first.c_str(), atof(pr.second.c_str()), maReplace);
      }
    }
    VSNode *out = nullptr;
    if (nofilter) {
      out = src; // mesure de plancher : core + source seule, aucun filtre GPU
    } else {
      vsapi->freeNode(src);
      VSMap *o;
      if (stdcopy) { // std.Invert : images de sortie allouees, zero GPU
        VSMap *ca = vsapi->createMap();
        vsapi->mapSetNode(ca, "clip", src, maReplace);
        o = vsapi->invoke(stdp, "Invert", ca);
        vsapi->freeMap(ca);
      } else {
        o = vsapi->invoke(plug, have_ref ? reffn.c_str() : "FFT3D", a);
      }
      if (vsapi->mapGetError(o)) { fprintf(stderr, "FFT3D: %s\n", vsapi->mapGetError(o)); return 1; }
      int err = 0;
      out = vsapi->mapGetNode(o, "clip", 0, &err);
      if (err || !out) { fprintf(stderr, "no clip\n"); return 1; }
      vsapi->freeMap(o);
    }
    vsapi->freeMap(a);

    char ebuf[512] = {};
    size_t m_nodes = priv_bytes();
    const int nfetch = fetch_override >= 0 ? fetch_override : g_frames;
    unsigned long long fnv = 1469598103934665603ULL;
    if (!nofetch)
      for (int n = fetch_start; n < fetch_start + nfetch; n++) {
        fprintf(stderr, "[harness] fetch %d\n", n);
        const VSFrame *f = vsapi->getFrame(n, out, ebuf, sizeof(ebuf));
        if (!f) { fprintf(stderr, "getFrame %d: %s\n", n, ebuf); return 1; }
        for (int pl = 0; pl < g_nplanes; pl++) {
          const unsigned char* fp = vsapi->getReadPtr(f, pl);
          ptrdiff_t st = vsapi->getStride(f, pl);
          int pw = pl == 0 ? g_w : g_w >> g_ssw;
          int ph = pl == 0 ? g_h : g_h >> g_ssh;
          for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw * g_bpp; x++) { fnv ^= fp[(size_t)y * st + x]; fnv *= 1099511628211ULL; }
        }
        fprintf(stderr, "[harness] got %d\n", n);
        vsapi->freeFrame(f);
      }
    fprintf(stderr, "[harness] fetched\n");
    size_t m_fetch = priv_bytes();
    vsapi->freeNode(out);
    fprintf(stderr, "[harness] node freed\n");
    size_t m_pre = priv_bytes();
    if (!onecore) vsapi->freeCore(core);
    fprintf(stderr, "[harness] core freed\n");
    if (vmmap && c == cycles - 1) {
      // histogram of committed private regions: shows WHERE the residue lives
      std::map<size_t, size_t> by_size; // region size -> total bytes
      MEMORY_BASIC_INFORMATION mbi;
      unsigned char* addr = nullptr;
      while (VirtualQuery(addr, &mbi, sizeof(mbi))) {
        if (mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE)
          by_size[mbi.RegionSize] += mbi.RegionSize;
        addr = (unsigned char*)mbi.BaseAddress + mbi.RegionSize;
      }
      fprintf(stderr, "[vmmap] committed private regions (Mo):\n");
      for (auto& kv2 : by_size)
        fprintf(stderr, "  region %8.1f Mo x %zu\n", kv2.first / 1048576.0, kv2.second / kv2.first);
    }
    size_t now = priv_bytes();
    printf("%4d  %10.1f  %+9.1f  fnv=%016llx  (load: %.1f nodes: %.1f fetch: %.1f freed: %.1f corefree: %.1f)\n", c + 1,
           now / 1048576.0, ((double)now - (double)prev) / 1048576.0, fnv,
           (m_load - m_core) / 1048576.0, (m_nodes - m_load) / 1048576.0, (m_fetch - m_nodes) / 1048576.0,
           ((double)m_pre - (double)m_fetch) / 1048576.0, ((double)now - (double)m_pre) / 1048576.0);
    fflush(stdout);
    prev = now;
  }
  return 0;
}
