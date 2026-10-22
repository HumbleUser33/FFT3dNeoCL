// plugin_smoke: loads the VapourSynth core as a library, feeds a raw fixture
// as a source filter, calls fft3dneocl.FFT3D, and writes the filtered planes
// to a raw file for comparison against the CPU harness output (out.raw).
// Usage: plugin_smoke <input.raw> <ref_out.raw> <W> <H> <bits> <fmt> <out.raw> [key=val ...]
//   fmt: GRAY | YUV420 | YUV422 | YUV444 | RGB
//   key=val: plugin args (bt=1 bw=16 sigma=2 planes=0 ...)

#include <windows.h>
#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

static std::vector<unsigned char> g_raw; // concatenated packed planes, per frame
static int g_w = 0, g_h = 0, g_bpp = 1, g_ssw = 0, g_ssh = 0, g_nplanes = 1;
static size_t g_framebytes = 0; // one packed frame; input.raw may hold several frames
static VSVideoFormat g_fmt {};

static const VSFrame *VS_CC srcGetFrame(int n, int ar, void *, void **,
                                        VSFrameContext *, VSCore *core, const VSAPI *vsapi) {
  if (ar != arInitial && ar != arAllFramesReady) return nullptr;
  VSFrame *f = vsapi->newVideoFrame(&g_fmt, g_w, g_h, nullptr, core);
  size_t base = (size_t)n * g_framebytes;
  if (base + g_framebytes > g_raw.size()) base = 0; // file shorter than the clip: repeat frame 0
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

static std::vector<unsigned char> read_file(const char* p, size_t* sz = nullptr) {
  FILE* f = fopen(p, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> v(n);
  if (fread(v.data(), 1, n, f) != n) { fprintf(stderr, "read error\n"); exit(1); }
  fclose(f);
  if (sz) *sz = n;
  return v;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    fprintf(stderr, "usage: plugin_smoke <input.raw> <ref_out.raw> W H bits fmt out.raw [key=val ...]\n");
    return 1;
  }
  const char* inpath = argv[1];
  const char* refpath = argv[2];
  int W = atoi(argv[3]), H = atoi(argv[4]), bits = atoi(argv[5]);
  std::string fmt = argv[6];
  const char* outpath = argv[7];

  int cf = cfGray, ssw = 0, ssh = 0, npl = 1;
  if (fmt == "GRAY") {}
  else if (fmt == "YUV420") { cf = cfYUV; ssw = 1; ssh = 1; npl = 3; }
  else if (fmt == "YUV422") { cf = cfYUV; ssw = 1; ssh = 0; npl = 3; }
  else if (fmt == "YUV444") { cf = cfYUV; ssw = 0; ssh = 0; npl = 3; }
  else if (fmt == "RGB")    { cf = cfRGB; npl = 3; }
  else { fprintf(stderr, "unknown format %s\n", fmt.c_str()); return 1; }

  g_raw = read_file(inpath);
  // refpath "-" means: no CPU reference available (timing-only runs)
  std::vector<unsigned char> ref;
  if (std::string(refpath) != "-")
    ref = read_file(refpath);
  g_w = W; g_h = H; g_ssw = ssw; g_ssh = ssh; g_nplanes = npl;
  g_bpp = bits <= 8 ? 1 : (bits <= 16 ? 2 : 4);
  g_framebytes = 0;
  for (int p = 0; p < npl; p++) {
    int pw = p == 0 ? W : W >> ssw;
    int ph = p == 0 ? H : H >> ssh;
    g_framebytes += (size_t)pw * ph * g_bpp;
  }

  // ---- load the core ----
  const char* VSDLL = "C:\\Users\\Bruno\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\vapoursynth\\libvapoursynth.dll";
  HMODULE vs = LoadLibraryA(VSDLL);
  if (!vs) { fprintf(stderr, "LoadLibrary VapourSynth failed: %lu\n", GetLastError()); return 1; }
  auto getAPI = (VSGetVapourSynthAPI)GetProcAddress(vs, "getVapourSynthAPI");
  if (!getAPI) { fprintf(stderr, "getVapourSynthAPI not found\n"); return 1; }
  const VSAPI* vsapi = getAPI(VS_MAKE_VERSION(4, 3));
  if (!vsapi) { fprintf(stderr, "API 4.3 not available\n"); return 1; }
  VSCore* core = vsapi->createCore(0);
  if (!core) { fprintf(stderr, "createCore failed\n"); return 1; }

  // ---- load our plugin ----
  VSPlugin* stdp = vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core);
  {
    VSMap* a = vsapi->createMap();
    const char* dll = "C:\\Users\\Bruno\\Desktop\\OpenCode\\FFT3DNEOCL\\fft3dneocl\\build\\Release\\fft3dneocl.dll";
    vsapi->mapSetData(a, "path", dll, -1, dtUtf8, maReplace);
    VSMap* o = vsapi->invoke(stdp, "LoadPlugin", a);
    if (vsapi->mapGetError(o)) { fprintf(stderr, "LoadPlugin: %s\n", vsapi->mapGetError(o)); return 1; }
    vsapi->freeMap(a); vsapi->freeMap(o);
  }
  VSPlugin* mine = vsapi->getPluginByID("com.github.fft3dneocl", core);
  if (!mine) { fprintf(stderr, "plugin not loaded\n"); return 1; }

  // ---- collect key=val args (special: frames, ref, reffn, refns, time) ----
  std::vector<std::pair<std::string, std::string>> kv;
  for (int i = 8; i < argc; i++) {
    std::string s = argv[i];
    size_t eq = s.find('=');
    if (eq != std::string::npos) kv.push_back({s.substr(0, eq), s.substr(eq + 1)});
  }
  auto take_special = [&](const char* k, std::string* out) {
    for (size_t i = 0; i < kv.size(); i++)
      if (kv[i].first == k) { if (out) *out = kv[i].second; kv.erase(kv.begin() + i); return true; }
    return false;
  };
  int frames = 1;
  { std::string v; if (take_special("frames", &v)) frames = atoi(v.c_str()); if (frames < 1) frames = 1; }
  // grab order stress-tests the plugin's fftcache: results must not depend on it
  std::string grab_order = "seq";
  take_special("order", &grab_order);
  std::string refdll, reffn = "FFT3DGPU", refns = "fft3dgpu";
  bool have_ref = take_special("ref", &refdll);
  take_special("reffn", &reffn);
  take_special("refns", &refns);
  std::string refout; // optional: write the ref filter's full output for a full comparison
  bool have_refout = take_special("refout", &refout);
  bool do_time = take_special("time", nullptr);
  // noplugin: benchmark/compare the ref filter alone, without creating a
  // fft3dneocl instance (clean memory and timing numbers for the ref)
  bool no_mine = take_special("noplugin", nullptr);

  // ---- source node from the raw fixture ----
  VSVideoInfo vi {};
  vi.width = W; vi.height = H; vi.numFrames = frames; vi.fpsNum = 24; vi.fpsDen = 1;
  if (!vsapi->queryVideoFormat(&vi.format, cf, bits == 32 ? stFloat : stInteger, bits, ssw, ssh, core)) {
    fprintf(stderr, "queryVideoFormat failed\n"); return 1;
  }
  g_fmt = vi.format;
  VSNode* src = vsapi->createVideoFilter2("RawSrc", &vi, srcGetFrame, srcFree, fmParallel, nullptr, 0, nullptr, core);
  if (!src) { fprintf(stderr, "source node failed\n"); return 1; }

  // build an invoke map; is_ref skips args the original fft3dgpu does not know
  auto build_args = [&](bool is_ref) {
    VSMap* a = vsapi->createMap();
    vsapi->mapSetNode(a, "clip", src, maReplace);
    for (auto& pr : kv) {
      if (is_ref) {
        static const char* ref_skip[] = {"planes","device_id","platform_id","num_streams",
          "l","t","r","b","opt","ncpu","mt","measure","pframe","px","py","pshow",
          "pcutoff","pfactor","hr","ht","dehalo"};
        bool skip = false;
        for (const char* s : ref_skip) if (pr.first == s) { skip = true; break; }
        if (skip) continue;
      } else {
        // args only the original fft3dgpu knows
        static const char* ours_skip[] = {"precision","mode","oldfft","plane","bordersize","device"};
        bool skip = false;
        for (const char* s : ours_skip) if (pr.first == s) { skip = true; break; }
        if (skip) continue;
      }
      const std::string& v = pr.second;
      // VapourSynth does not coerce int->float, so force the right type by name
      static const char* float_names[] = {"sigma","sigma2","sigma3","sigma4","beta","sharpen",
        "scutoff","svr","smin","smax","kratio","degrid","pcutoff","pfactor","hr","ht","dehalo"};
      bool isfloat = false;
      for (const char* s : float_names) if (pr.first == s) { isfloat = true; break; }
      bool isint = !isfloat && !v.empty();
      for (size_t j = (v[0] == '-' ? 1 : 0); j < v.size(); j++) if (!isdigit((unsigned char)v[j])) isint = false;
      if (pr.first == "planes") {
        std::vector<int64_t> pv;
        size_t pos = 0;
        while (pos < v.size()) {
          size_t c = v.find(',', pos);
          pv.push_back(atoll(v.substr(pos, c == std::string::npos ? std::string::npos : c - pos).c_str()));
          if (c == std::string::npos) break;
          pos = c + 1;
        }
        vsapi->mapSetIntArray(a, pr.first.c_str(), pv.data(), (int)pv.size());
      } else if (isint) {
        vsapi->mapSetInt(a, pr.first.c_str(), atoll(v.c_str()), maReplace);
      } else {
        vsapi->mapSetFloat(a, pr.first.c_str(), atof(v.c_str()), maReplace);
      }
    }
    return a;
  };

  VSNode* out;
  if (no_mine) {
    out = src; // ref-only run: the source node is the "output" (comparisons skipped)
  } else {
    VSMap* a = build_args(false);
    vsapi->freeNode(src); // mapSetNode took its own reference(s)
    VSMap* o = vsapi->invoke(mine, "FFT3D", a);
    if (vsapi->mapGetError(o)) { fprintf(stderr, "FFT3D: %s\n", vsapi->mapGetError(o)); return 1; }
    int err = 0;
    out = vsapi->mapGetNode(o, "clip", 0, &err);
    if (err || !out) { fprintf(stderr, "FFT3D returned no clip\n"); return 1; }
    vsapi->freeMap(a); vsapi->freeMap(o);
  }

  // fetch frame n as packed planes
  auto grab = [&](VSNode* node, int n) {
    char ebuf[512] = {};
    const VSFrame* f = vsapi->getFrame(n, node, ebuf, sizeof(ebuf));
    if (!f) { fprintf(stderr, "getFrame %d: %s\n", n, ebuf); exit(1); }
    std::vector<unsigned char> v;
    for (int p = 0; p < npl; p++) {
      int pw = p == 0 ? W : W >> ssw;
      int ph = p == 0 ? H : H >> ssh;
      ptrdiff_t st = vsapi->getStride(f, p);
      const uint8_t* s = vsapi->getReadPtr(f, p);
      for (int y = 0; y < ph; y++)
        v.insert(v.end(), s + (size_t)y * st, s + (size_t)y * st + (size_t)pw * g_bpp);
    }
    vsapi->freeFrame(f);
    return v;
  };

  // ---- get every frame, write packed planes ----
  std::vector<std::vector<unsigned char>> gotf(frames);
  {
    std::vector<int> order(frames);
    for (int n = 0; n < frames; n++) order[n] = n;
    if (grab_order == "rev")
      std::reverse(order.begin(), order.end());
    else if (grab_order == "shuf") {
      std::mt19937 rng(12345); // fixed seed: reproducible stress test
      std::shuffle(order.begin(), order.end(), rng);
    }
    for (int n : order) gotf[n] = grab(out, n);
  }
  if (!no_mine) {
    FILE* fo = fopen(outpath, "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", outpath); return 1; }
    for (auto& g : gotf) fwrite(g.data(), 1, g.size(), fo);
    fclose(fo);
  }
  std::vector<unsigned char> got;
  for (auto& g : gotf) got.insert(got.end(), g.begin(), g.end());

  // generic comparator (integer or float)
  auto cmp = [&](const std::vector<unsigned char>& A, const std::vector<unsigned char>& B, const char* name) {
    if (A.size() != B.size()) { fprintf(stderr, "%s size mismatch\n", name); return; }
    long nd = 0; int mx = 0; float fmx = 0.0f;
    if (g_bpp == 4) {
      const float* x = (const float*)A.data(); const float* y = (const float*)B.data();
      size_t n = A.size() / 4;
      for (size_t i = 0; i < n; i++) {
        if (memcmp(&x[i], &y[i], 4) != 0) nd++;
        float d = std::abs(x[i] - y[i]);
        if (d > fmx) fmx = d;
      }
      printf("%-24s ndiff=%ld (%.2f%%) maxabs=%.4g  [%zu floats]\n", name, nd, 100.0 * nd / n, fmx, n);
    } else {
      for (size_t i = 0; i < A.size(); i++) {
        int d = std::abs((int)A[i] - (int)B[i]);
        if (d) nd++;
        if (d > mx) mx = d;
      }
      printf("%-24s ndiff=%ld (%.2f%%) maxabs=%d  [%zu bytes]\n", name, nd, 100.0 * nd / A.size(), mx, A.size());
    }
  };

  if (!ref.empty() && !no_mine) {
    // compare as many frames as the reference holds
    size_t fb = gotf[0].size();
    size_t nf = ref.size() / fb;
    if (nf > (size_t)frames) nf = frames;
    if (nf == 0) nf = 1;
    printf("PLUGIN vs harness:       (%zu frame(s) compared)\n", nf);
    cmp(std::vector<unsigned char>(got.begin(), got.begin() + nf * fb),
        std::vector<unsigned char>(ref.begin(), ref.begin() + nf * fb),
        "  total:");
  } else {
    printf("PLUGIN vs harness:       (no reference given)\n");
  }

  // ---- optional: original fft3dgpu (DX9) on the same frame ----
  VSNode* refnode = nullptr;
  if (have_ref) {
    VSMap* la = vsapi->createMap();
    vsapi->mapSetData(la, "path", refdll.c_str(), -1, dtUtf8, maReplace);
    VSMap* lo = vsapi->invoke(stdp, "LoadPlugin", la);
    if (vsapi->mapGetError(lo)) { fprintf(stderr, "LoadPlugin ref: %s\n", vsapi->mapGetError(lo)); return 1; }
    vsapi->freeMap(la); vsapi->freeMap(lo);
    VSPlugin* rp = vsapi->getPluginByNamespace(refns.c_str(), core);
    if (!rp) { fprintf(stderr, "ref plugin ns '%s' not found\n", refns.c_str()); return 1; }
    VSMap* ra = build_args(true);
    VSMap* ro = vsapi->invoke(rp, reffn.c_str(), ra);
    if (vsapi->mapGetError(ro)) { fprintf(stderr, "%s: %s\n", reffn.c_str(), vsapi->mapGetError(ro)); return 1; }
    int e2 = 0;
    refnode = vsapi->mapGetNode(ro, "clip", 0, &e2);
    if (e2 || !refnode) { fprintf(stderr, "%s returned no clip\n", reffn.c_str()); return 1; }
    vsapi->freeMap(ra); vsapi->freeMap(ro);
    std::vector<unsigned char> gpu = grab(refnode, 0);
    if (!no_mine) cmp(gotf[0], gpu, "PLUGIN vs fft3dgpu:");
    if (!ref.empty()) {
      size_t fb = gotf[0].size();
      cmp(gpu, std::vector<unsigned char>(ref.begin(), ref.begin() + fb), "fft3dgpu vs harness:");
    }
    if (have_refout) {
      // full-clip quality comparison of the reference filter
      std::vector<unsigned char> reffull;
      for (int n = 0; n < frames; n++) {
        std::vector<unsigned char> g = grab(refnode, n);
        reffull.insert(reffull.end(), g.begin(), g.end());
      }
      FILE* fr = fopen(refout.c_str(), "wb");
      if (!fr) { fprintf(stderr, "cannot write %s\n", refout.c_str()); return 1; }
      fwrite(reffull.data(), 1, reffull.size(), fr);
      fclose(fr);
      if (!no_mine) cmp(got, reffull, "PLUGIN vs REF:");
      size_t m = reffull.size() < ref.size() ? reffull.size() : ref.size();
      if (m) cmp(std::vector<unsigned char>(reffull.begin(), reffull.begin() + m),
                 std::vector<unsigned char>(ref.begin(), ref.begin() + m), "REF vs harness:");
    }
  }

  // ---- optional: performance over N frames ----
  if (do_time && frames > 1) {
    auto bench = [&](VSNode* node, const char* name) {
      grab(node, 0); // warmup (GPU init, shader compile)
      // the earlier fetch loops left frames in VapourSynth's caches: without this
      // the timed loop would measure cache hits instead of real filter work
      vsapi->clearCoreCaches(core);
      auto t0 = std::chrono::steady_clock::now();
      for (int n = 0; n < frames; n++) grab(node, n);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      printf("perf %-18s: %d frames in %.1f ms -> %.2f ms/frame (%.1f fps)\n",
             name, frames, ms, ms / frames, frames * 1000.0 / ms);
    };
    if (!no_mine) bench(out, "fft3dneocl");
    if (refnode) bench(refnode, refns.c_str());
  }

  return 0;
}
