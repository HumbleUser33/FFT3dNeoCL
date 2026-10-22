// Standalone CPU reference harness: drives the ORIGINAL FFT3DEngine
// (neo_FFT3D sources, unmodified) with synthetic frames and writes raw
// planar output. This is the numeric oracle for the OpenCL port.
//
// Usage: fft3d_ref --out out.raw [options] (see print_usage)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

#include "fft3d_engine.h"

#include "../../tests/framegen.hpp"

static void print_usage() {
  printf(
    "fft3d_ref - standalone reference for neo_FFT3D\n"
    "  --out FILE            raw output (planes concatenated, no padding)\n"
    "  --in FILE             raw input (planes concatenated, no padding) instead of --seed\n"
    "  --width N --height N --frames N\n"
    "  --format GRAY|YUV420|YUV422|YUV444|RGBP\n"
    "  --bitdepth 8|10|12|14|16|32   (32 = float)\n"
    "  --seed N\n"
    "  --planes 0,1,2        planes to process (others copied)\n"
    "  FFT params: --sigma --beta --bw --bh --bt --ow --oh --kratio --sharpen\n"
    "    --scutoff --svr --smin --smax --measure 0/1 --interlaced 0/1 --wintype N\n"
    "    --pframe N --px N --py N --pshow 0/1 --pcutoff --pfactor\n"
    "    --sigma2 --sigma3 --sigma4 --degrid --dehalo --hr --ht\n"
    "    --l --t --r --b --opt N\n"
  );
}

struct Args {
  std::string out = "out.raw";
  std::string infile = "";
  std::string dumpdir = "";
  int width = 320, height = 240, frames = 10;
  std::string format = "YUV420";
  int bitdepth = 8;
  uint32_t seed = 12345;
  std::vector<int> planes {0, 1, 2};

  // FFT3D params, defaults identical to fft3d.hpp Initialize()
  float sigma = 2.0f, beta = 1.0f;
  int bw = 32, bh = 32, bt = 3, ow = -1, oh = -1;
  float kratio = 2.0f, sharpen = 0.0f, scutoff = 0.3f, svr = 1.0f, smin = 4.0f, smax = 20.0f;
  bool measure = true, interlaced = false;
  int wintype = 0, pframe = 0, px = 0, py = 0;
  bool pshow = false;
  float pcutoff = 0.1f, pfactor = 0.0f;
  float sigma2 = -1, sigma3 = -1, sigma4 = -1; // -1 => same as sigma (like the plugin)
  float degrid = 1.0f, dehalo = 0.0f, hr = 2.0f, ht = 50.0f;
  int l = 0, t = 0, r = 0, b = 0, opt = 0;
  bool do_time = false;
};

static bool is_float_depth(int bd) { return bd == 32; }

struct HarnessFrames {
  GenFormat gf;
  bool is_yuv {true};
  int width, height, frames;
  uint32_t seed;
  std::vector<std::vector<unsigned char>> pixels; // [frame*planes + plane]
  std::vector<std::vector<const unsigned char*>> ptrs; // per frame
  std::vector<std::vector<int>> strides; // per frame

  HarnessFrames(const GenFormat& g, bool yuv, int w, int h, int f, uint32_t s)
    : gf(g), is_yuv(yuv), width(w), height(h), frames(f), seed(s) {
    pixels.resize((size_t)f * g.Planes);
    ptrs.resize(f);
    strides.resize(f);
    for (int n = 0; n < f; n++) {
      ptrs[n].resize(g.Planes);
      strides[n].resize(g.Planes);
      for (int p = 0; p < g.Planes; p++) {
        int pw = p == 0 ? w : (w >> g.SSW);
        int ph = p == 0 ? h : (h >> g.SSH);
        gen_plane(gf, p, n, pw, ph, seed, pixels[(size_t)n * g.Planes + p]);
        ptrs[n][p] = pixels[(size_t)n * g.Planes + p].data();
        strides[n][p] = pw * gf.BytesPerSample;
      }
    }
  }

  DSFrame frame(int n) {
    if (n < 0) n = 0;
    if (n >= frames) n = frames - 1;
    DSFormat fmt;
    fmt.Planes = gf.Planes; fmt.SSW = gf.SSW; fmt.SSH = gf.SSH;
    fmt.BitsPerSample = gf.BitsPerSample; fmt.BytesPerSample = gf.BytesPerSample;
    fmt.IsFloat = gf.IsFloat; fmt.IsInteger = !gf.IsFloat;
    fmt.IsFamilyYUV = is_yuv; fmt.IsFamilyRGB = !is_yuv;
    return DSFrame(ptrs[n].data(), strides[n].data(), fmt, width, height);
  }
};

struct HarnessFetch : FetchFrameFunctor {
  HarnessFrames* store;
  DSFrame operator()(int n) override { return store->frame(n); }
};

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { print_usage(); std::exit(2); }
      return argv[++i];
    };
    auto nf = [&]() { return atoi(next()); };
    auto df = [&]() { return (float)atof(next()); };
    auto nb = [&]() { return atoi(next()) != 0; };
    if      (k == "--out") a.out = next();
    else if (k == "--in") a.infile = next();
    else if (k == "--width") a.width = nf();
    else if (k == "--height") a.height = nf();
    else if (k == "--frames") a.frames = nf();
    else if (k == "--format") a.format = next();
    else if (k == "--bitdepth") a.bitdepth = nf();
    else if (k == "--seed") a.seed = (uint32_t)atoi(next());
    else if (k == "--planes") {
      a.planes.clear();
      std::string s = next();
      const char* c = s.c_str();
      while (*c) { a.planes.push_back(atoi(c)); while (*c && *c != ',') c++; if (*c == ',') c++; }
    }
    else if (k == "--sigma") a.sigma = df();
    else if (k == "--beta") a.beta = df();
    else if (k == "--bw") a.bw = nf();
    else if (k == "--bh") a.bh = nf();
    else if (k == "--bt") a.bt = nf();
    else if (k == "--ow") a.ow = nf();
    else if (k == "--oh") a.oh = nf();
    else if (k == "--kratio") a.kratio = df();
    else if (k == "--sharpen") a.sharpen = df();
    else if (k == "--scutoff") a.scutoff = df();
    else if (k == "--svr") a.svr = df();
    else if (k == "--smin") a.smin = df();
    else if (k == "--smax") a.smax = df();
    else if (k == "--measure") a.measure = nb();
    else if (k == "--interlaced") a.interlaced = nb();
    else if (k == "--wintype") a.wintype = nf();
    else if (k == "--pframe") a.pframe = nf();
    else if (k == "--px") a.px = nf();
    else if (k == "--py") a.py = nf();
    else if (k == "--pshow") a.pshow = nb();
    else if (k == "--pcutoff") a.pcutoff = df();
    else if (k == "--pfactor") a.pfactor = df();
    else if (k == "--sigma2") a.sigma2 = df();
    else if (k == "--sigma3") a.sigma3 = df();
    else if (k == "--sigma4") a.sigma4 = df();
    else if (k == "--degrid") a.degrid = df();
    else if (k == "--dehalo") a.dehalo = df();
    else if (k == "--hr") a.hr = df();
    else if (k == "--ht") a.ht = df();
    else if (k == "--l") a.l = nf();
    else if (k == "--t") a.t = nf();
    else if (k == "--r") a.r = nf();
    else if (k == "--b") a.b = nf();
    else if (k == "--opt") a.opt = nf();
    else if (k == "--time") a.do_time = true;
    else if (k == "--dump-input") { /* handled after frame generation */ }
    else if (k == "--dump-stages") {
      a.dumpdir = next();
#ifdef _WIN32
      _putenv_s("FFT3D_DUMPDIR", a.dumpdir.c_str());
#else
      setenv("FFT3D_DUMPDIR", a.dumpdir.c_str(), 1);
#endif
    }
    else if (k == "--help") { print_usage(); return 0; }
    else { printf("unknown option: %s\n", k.c_str()); print_usage(); return 2; }
  }

  bool flt = is_float_depth(a.bitdepth);
  GenFormat gf = GenFormatFromName(a.format, a.bitdepth, flt);
  if (gf.Planes < 0) { printf("bad --format\n"); return 2; }

  DSFormat fmt;
  fmt.Planes = gf.Planes;
  fmt.SSW = gf.SSW; fmt.SSH = gf.SSH;
  fmt.BitsPerSample = gf.BitsPerSample; fmt.BytesPerSample = gf.BytesPerSample;
  fmt.IsFloat = gf.IsFloat; fmt.IsInteger = !gf.IsFloat;
  fmt.IsFamilyYUV = (a.format != "RGBP");
  fmt.IsFamilyRGB = (a.format == "RGBP");

  DSVideoInfo vi(fmt, 24000, 1001, a.width, a.height, a.frames);

  // EngineParams with the plugin's defaults (fft3d.hpp Initialize)
  float sigma1 = a.sigma;
  EngineParams ep {
    sigma1, a.beta,
    a.bw, a.bh,
    a.bt,
    a.ow, a.oh,
    a.kratio,
    a.sharpen,
    a.scutoff,
    a.svr,
    a.smin, a.smax,
    a.measure,
    a.interlaced,
    a.wintype,
    a.pframe,
    a.px, a.py,
    a.pshow,
    a.pcutoff,
    a.pfactor,
    a.sigma2 < 0 ? sigma1 : a.sigma2,
    a.sigma3 < 0 ? sigma1 : a.sigma3,
    a.sigma4 < 0 ? sigma1 : a.sigma4,
    a.degrid,
    a.dehalo,
    a.hr, a.ht,
    MAX(a.l, 0), MAX(a.t, 0), MAX(a.r, 0), MAX(a.b, 0),
    a.opt,
    vi,
    nullptr,
    false
  };

  if (ep.l + ep.r >= ep.vi.Width) { printf("crop error: width\n"); return 2; }
  if (ep.t + ep.b >= ep.vi.Height) { printf("crop error: height\n"); return 2; }

  int process[4] = {2, 2, 2, 2};
  for (int p : a.planes)
    if (p < fmt.Planes) process[p] = 3;

  HarnessFrames store(gf, fmt.IsFamilyYUV, a.width, a.height, a.frames, a.seed);

  if (!a.infile.empty()) {
    // real video frames: packed planes, no padding (same layout as --dump-input)
    size_t fb = 0;
    for (int p = 0; p < fmt.Planes; p++) {
      int pw = p == 0 ? a.width : (a.width >> fmt.SSW);
      int ph = p == 0 ? a.height : (a.height >> fmt.SSH);
      fb += (size_t)pw * ph * fmt.BytesPerSample;
    }
    FILE* fi = fopen(a.infile.c_str(), "rb");
    if (!fi) { printf("cannot open --in %s\n", a.infile.c_str()); return 1; }
    fseek(fi, 0, SEEK_END); size_t fsz = (size_t)ftell(fi); fseek(fi, 0, SEEK_SET);
    if (fsz < fb * (size_t)a.frames) {
      printf("--in too small: %zu bytes, need %zu for %d frames\n", fsz, fb * (size_t)a.frames, a.frames);
      fclose(fi); return 1;
    }
    std::vector<unsigned char> buf(fb);
    for (int n = 0; n < a.frames; n++) {
      if (fread(buf.data(), 1, fb, fi) != fb) { printf("--in read error\n"); fclose(fi); return 1; }
      size_t off = 0;
      for (int p = 0; p < fmt.Planes; p++) {
        int pw = p == 0 ? a.width : (a.width >> fmt.SSW);
        int ph = p == 0 ? a.height : (a.height >> fmt.SSH);
        size_t pb = (size_t)pw * ph * fmt.BytesPerSample;
        memcpy(store.pixels[(size_t)n * fmt.Planes + p].data(), buf.data() + off, pb);
        off += pb;
      }
    }
    fclose(fi);
    printf("loaded %d frames from %s\n", a.frames, a.infile.c_str());
  }

  if (argc > 1) {
    for (int i = 1; i < argc; i++)
      if (!strcmp(argv[i], "--dump-input")) {
        FILE* fi = fopen(a.out.c_str(), "wb");
        for (int n = 0; n < a.frames; n++)
          for (int p = 0; p < fmt.Planes; p++) {
            int pw = (p == 0 ? a.width : (a.width >> fmt.SSW)) * fmt.BytesPerSample;
            int ph = p == 0 ? a.height : (a.height >> fmt.SSH);
            fwrite(store.pixels[(size_t)n * fmt.Planes + p].data(), 1, (size_t)pw * ph, fi);
          }
        fclose(fi);
        {
          std::string metafile = a.out + ".meta.txt";
          FILE* fm = fopen(metafile.c_str(), "wb");
          fprintf(fm, "width=%d\nheight=%d\nplanes=%d\nssw=%d\nssh=%d\nbitdepth=%d\nbytespersample=%d\nfloat=%d\nframes=%d\nseed=%u\n",
                  a.width, a.height, fmt.Planes, fmt.SSW, fmt.SSH, fmt.BitsPerSample, fmt.BytesPerSample,
                  fmt.IsFloat ? 1 : 0, a.frames, a.seed);
          fclose(fm);
        }
        printf("OK wrote input %s\n", a.out.c_str());
        return 0;
      }
  }

  HarnessFetch fetch;
  fetch.store = &store;

  FFTFunctionPointers fftfp;
  try {
    fftfp.load();
  } catch (const char* e) { printf("fftw load error: %s\n", e); return 1; }

  FFT3DEngine* engine[4] = {nullptr, nullptr, nullptr, nullptr};
  int engine_count = 0, copy_count = 0;
  try {
    for (int i = 0; i < fmt.Planes; i++) {
      if (process[i] == 3) {
        ep.IsChroma = fmt.IsFamilyYUV && i != 0;
        ep.framewidth = ep.IsChroma ? (a.width >> fmt.SSW) : a.width;
        ep.frameheight = ep.IsChroma ? (a.height >> fmt.SSH) : a.height;
        engine[i] = new FFT3DEngine(ep, i, &fetch, fftfp);
        engine_count++;
      } else if (process[i] == 2) {
        copy_count++;
      }
    }
  } catch (const char* e) {
    printf("engine init error: %s\n", e);
    return 1;
  }

  bool crop = ep.l > 0 || ep.r > 0 || ep.t > 0 || ep.b > 0;

  FILE* fout = fopen(a.out.c_str(), "wb");
  if (!fout) { printf("cannot open %s\n", a.out.c_str()); return 1; }

  std::unordered_map<int, DSFrame> empty_frames;

  // --time: measure only the engine (excludes frame generation and file I/O)
  double eng_ms = 0.0;
  auto call_engine = [&](int i, int n) -> DSFrame {
    auto t0 = std::chrono::steady_clock::now();
    DSFrame f = engine[i]->GetFrame(n, empty_frames);
    eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return f;
  };

  for (int n = 0; n < a.frames; n++) {
    DSFrame src = store.frame(n);
    DSFrame dst;

    if (engine_count == 1 && copy_count == 0 && !crop) {
      // plugin fast path: single engine, no copies, no crop
      dst = call_engine(0, n);
    } else {
      dst = src.Create(false);
      // mark untouched planes (process==1) with a sentinel for debugging
      for (int p = 0; p < fmt.Planes; p++)
        if (process[p] == 1) {
          int pw = p == 0 ? a.width : (a.width >> fmt.SSW);
          int ph = p == 0 ? a.height : (a.height >> fmt.SSH);
          memset(dst.DstPointers[p], 0xAA, (size_t)dst.StrideBytes[p] * ph);
        }
      for (int i = 0; i < fmt.Planes; i++) {
        bool chroma = fmt.IsFamilyYUV && i > 0 && i < 3;
        if (process[i] == 3) {
          int l = chroma ? (ep.l >> fmt.SSW) : ep.l;
          int r = chroma ? (ep.r >> fmt.SSW) : ep.r;
          int t = chroma ? (ep.t >> fmt.SSH) : ep.t;
          int b = chroma ? (ep.b >> fmt.SSH) : ep.b;
          DSFrame frame = call_engine(i, n);
          const unsigned char* srcp = src.SrcPointers[i];
          const unsigned char* pcs = frame.SrcPointers ? frame.SrcPointers[i] : nullptr;
          unsigned char* dstp = dst.DstPointers[i];
          int ss = src.StrideBytes[i], ds = dst.StrideBytes[i], ps = frame.StrideBytes ? frame.StrideBytes[i] : 0;
          int pw = (chroma ? (a.width >> fmt.SSW) : a.width) * fmt.BytesPerSample;
          int ph = chroma ? (a.height >> fmt.SSH) : a.height;
          if (!pcs) pcs = srcp; // engine returned source (e.g. Kalman frame 0)
          int t2 = t, b2 = b, l2 = l * fmt.BytesPerSample, r2 = r * fmt.BytesPerSample;
          if (t2 > 0) { memcpy(dstp, srcp, (size_t)ds * t2); srcp += (size_t)ss * t2; dstp += (size_t)ds * t2; pcs += (size_t)ps * t2; }
          for (int y = 0; y < ph - t2 - b2; y++) {
            if (l2 > 0) memcpy(dstp, srcp, l2);
            memcpy(dstp + l2, pcs + l2, pw - l2 - r2);
            if (r2 > 0) memcpy(dstp + pw - r2, srcp + pw - r2, r2);
            srcp += ss; dstp += ds; pcs += ps;
          }
          if (b2 > 0) memcpy(dstp, srcp, (size_t)ds * b2);
        } else if (process[i] == 2) {
          int ph = (i == 0 ? a.height : (a.height >> fmt.SSH));
          memcpy(dst.DstPointers[i], src.SrcPointers[i], (size_t)src.StrideBytes[i] * ph);
        }
      }
    }

    // write planes, no padding
    const unsigned char* wptr = dst.DstPointers ? dst.DstPointers[0] : nullptr;
    for (int p = 0; p < fmt.Planes; p++) {
      const unsigned char* rp = (dst.DstPointers ? dst.DstPointers[p] : dst.SrcPointers[p]);
      int pw = (p == 0 ? a.width : (a.width >> fmt.SSW)) * fmt.BytesPerSample;
      int ph = p == 0 ? a.height : (a.height >> fmt.SSH);
      int st = dst.StrideBytes[p];
      for (int y = 0; y < ph; y++)
        fwrite(rp + (size_t)y * st, 1, pw, fout);
    }
    (void)wptr;
  }

  fclose(fout);

  if (a.do_time && a.frames > 0 && eng_ms > 0.0)
    printf("cpu engine: %.2f ms/frame (%.1f fps) over %d frames\n",
           eng_ms / a.frames, a.frames * 1000.0 / eng_ms, a.frames);

  // meta file for the comparator
  std::string metafile = a.out + ".meta.txt";
  FILE* fm = fopen(metafile.c_str(), "wb");
  fprintf(fm, "width=%d\nheight=%d\nplanes=%d\nssw=%d\nssh=%d\nbitdepth=%d\nbytespersample=%d\nfloat=%d\nframes=%d\nseed=%u\n",
          a.width, a.height, fmt.Planes, fmt.SSW, fmt.SSH, fmt.BitsPerSample, fmt.BytesPerSample,
          fmt.IsFloat ? 1 : 0, a.frames, a.seed);
  fclose(fm);

  for (int i = 0; i < 4; i++)
    if (engine[i]) delete engine[i];
  fftfp.free();
  printf("OK wrote %s (%d frames)\n", a.out.c_str(), a.frames);
  return 0;
}
