// fft3dneocl — VapourSynth API4 plugin shell for the OpenCL port of neo_FFT3D.
// Parameter names, order and defaults follow neo_FFT3D/src/fft3d.hpp Params()
// (see docs/params.md). GPU-specific additions: device_id, platform_id,
// num_streams (accepted; output is bit-identical regardless of their value).
//
// Current scope: bt=-1 (sharpen only), bt=0 (Kalman, fmUnordered — the API4
// replacement for the original's fmSerial), bt=1..5 (2D and temporal Wiener)
// and pattern denoise (pfactor != 0, including the sigma2/3/4 variant).
// pshow (debug visualization) returns a clear error.

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ocl/ocl_loader.hpp"
#include "ocl/context.hpp"
#include "gpu/tables.hpp"
#include "gpu/pipeline.hpp"
#include "kernel_source_gen.hpp" // FFT3D_CL_SRC (generated from src/kernels/fft3d.cl)

namespace {

struct FFT3DData {
  VSNode *child = nullptr;
  VSVideoInfo vi {};
  int bt = 1;                                          // temporal window size
  std::vector<std::shared_ptr<gpu::GpuPipeline>> pipes; // one per plane, null = copy
  int cl[4] = {}, ct[4] = {};                           // crop offset per plane (pixels)
  bool crop = false;
  // pattern mode: the engine fetches frame pframe once per instance
  // (fft3d_engine.h 543-552); the sigmas variant builds the pattern without
  // a frame (engine 373-378), so no extra request is needed in that case.
  bool needPatFrame = false;
  int pframe = 0;
  // VapourSynth may call getFrame for the same node from several threads, and
  // the pipelines share their GPU buffers, so the filter body is serialized.
  std::mutex mtx;
};

bool isPowerOfTwo16_128(int v) {
  return (v == 16 || v == 32 || v == 64 || v == 128);
}

const VSFrame *VS_CC fft3dGetFrame(int n, int activationReason, void *instanceData, void **frameData,
                                   VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi);
void VS_CC fft3dFree(void *instanceData, VSCore *core, const VSAPI *vsapi);

void VS_CC fft3dCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
  int err = 0;
  VSNode *clip = nullptr;
  // mapGetNode adds a reference: every failure path after it must freeNode
  auto fail = [&](const char* msg) {
    vsapi->mapSetError(out, msg);
    if (clip) vsapi->freeNode(clip);
  };

  clip = vsapi->mapGetNode(in, "clip", 0, &err);
  if (err || !clip) { clip = nullptr; fail("fft3dneocl: clip argument missing"); return; }
  const VSVideoInfo *vi = vsapi->getVideoInfo(clip);
  if (!vsh::isConstantVideoFormat(vi)) { fail("fft3dneocl: only constant format clips are supported"); return; }

  // ---- parameters (original names, original defaults) ----
  auto iarg = [&](const char* k, int def) {
    if (vsapi->mapNumElements(in, k) <= 0) return def;
    int e = 0; int64_t v = vsapi->mapGetInt(in, k, 0, &e);
    return e ? def : (int)v;
  };
  auto farg = [&](const char* k, double def) {
    if (vsapi->mapNumElements(in, k) <= 0) return (float)def;
    int e = 0; double v = vsapi->mapGetFloat(in, k, 0, &e);
    return e ? (float)def : (float)v;
  };

  float sigma = farg("sigma", 2.0);
  float beta = farg("beta", 1.0);
  int bw = iarg("bw", 32), bh = iarg("bh", 32);
  int bt = iarg("bt", 3);
  int ow = iarg("ow", -1), oh = iarg("oh", -1);
  float kratio = farg("kratio", 2.0);
  float sharpen = farg("sharpen", 0.0);
  float scutoff = farg("scutoff", 0.3);
  float svr = farg("svr", 1.0);
  float smin = farg("smin", 4.0), smax = farg("smax", 20.0);
  // measure: accepted, no effect on GPU
  int interlaced = iarg("interlaced", 0);
  int wintype = iarg("wintype", 0);
  // pframe/px/py: pattern source frame and fixed pattern block
  int pframe = iarg("pframe", 0), px = iarg("px", 0), py = iarg("py", 0);
  int pshow = iarg("pshow", 0);
  float pcutoff = farg("pcutoff", 0.1);
  float pfactor = farg("pfactor", 0.0);
  float sigma2 = farg("sigma2", sigma), sigma3 = farg("sigma3", sigma), sigma4 = farg("sigma4", sigma);
  float degrid = farg("degrid", 1.0);
  float dehalo = farg("dehalo", 0.0);
  float hr = farg("hr", 2.0), ht = farg("ht", 50.0);
  int l = iarg("l", 0), t = iarg("t", 0), r = iarg("r", 0), b = iarg("b", 0);
  // opt/ncpu/mt: accepted, no effect on GPU
  int device_id = iarg("device_id", 0);
  int platform_id = iarg("platform_id", -1);
  // num_streams: accepted, output must be bit-identical (single stream for now)

  // ---- planes to process (original VS default: all) ----
  int planes[3] = {0, 1, 2};
  int nplanes_sel = vi->format.numPlanes; // default: all planes
  int np = vsapi->mapNumElements(in, "planes");
  if (np > 0) {
    const int64_t *pa = vsapi->mapGetIntArray(in, "planes", &err);
    if (err || !pa) { fail("fft3dneocl: bad planes argument"); return; }
    nplanes_sel = np < 3 ? np : 3;
    for (int i = 0; i < nplanes_sel; i++) planes[i] = (int)pa[i];
  }

  // ---- validation (same order/messages as the original where possible) ----
  if (bt < -1 || bt > 5) {
    fail("bt must be -1 (Sharpen), 0 (Kalman), 1..5 (Wiener)");
    return;
  }
  if (pshow) { fail("fft3dneocl (GPU port): pshow is not implemented"); return; }
  if (beta < 1.0f) { fail("beta must be not less 1.0"); return; }
  if (!isPowerOfTwo16_128(bw) || !isPowerOfTwo16_128(bh)) {
    fail("fft3dneocl (GPU port): bw and bh must be powers of two in [16, 128]");
    return;
  }
  if (ow * 2 > bw) { fail("ow must be less than bw / 2"); return; }
  if (oh * 2 > bh) { fail("oh must be less than bh / 2"); return; }
  if (wintype < 0 || wintype > 2) { fail("fft3dneocl: wintype must be 0, 1 or 2"); return; }
  if (interlaced < 0 || interlaced > 1) { fail("fft3dneocl: interlaced must be 0 or 1"); return; }

  int bits = vi->format.bitsPerSample, bpp = vi->format.bytesPerSample;
  if (vi->format.sampleType == stInteger) {
    if (bits != 8 && bits != 10 && bits != 12 && bits != 14 && bits != 16) {
      fail("fft3dneocl: integer clips must be 8/10/12/14/16 bits"); return;
    }
  } else if (vi->format.sampleType == stFloat) {
    if (bits != 32) { fail("fft3dneocl: float clips must be 32-bit (float16 not supported)"); return; }
  } else { fail("fft3dneocl: unknown sample type"); return; }
  if (vi->format.colorFamily != cfGray && vi->format.colorFamily != cfYUV && vi->format.colorFamily != cfRGB) {
    fail("fft3dneocl: only Gray, YUV and planar RGB are supported"); return;
  }

  if (l < 0) l = 0; if (t < 0) t = 0; if (r < 0) r = 0; if (b < 0) b = 0;
  if (l + r >= vi->width) { fail("Width cannot be cropped to zero or below"); return; }
  if (t + b >= vi->height) { fail("Height cannot be cropped to zero or below"); return; }

  // ---- build one GPU pipeline per processed plane ----
  auto data = std::make_unique<FFT3DData>();
  data->child = clip;
  data->vi = *vi;
  data->bt = bt;
  data->crop = (l > 0 || r > 0 || t > 0 || b > 0);
  data->pipes.resize(vi->format.numPlanes);
  // engine 543: any pfactor != 0 estimates the pattern from frame pframe on the
  // first GetFrame call (the sigmas variant already set it in the pipeline ctor)
  data->needPatFrame = (pfactor != 0.0f);
  data->pframe = pframe;

  for (int plane = 0; plane < vi->format.numPlanes; plane++) {
    bool process = false;
    for (int i = 0; i < nplanes_sel; i++) if (planes[i] == plane) process = true;
    if (!process) continue;

    bool chroma = (vi->format.colorFamily == cfYUV && plane > 0);
    gpu::FilterParams fp;
    fp.sigma = sigma; fp.beta = beta;
    fp.bw = bw; fp.bh = bh; fp.bt = bt;
    fp.ow = ow; fp.oh = oh;
    fp.kratio = kratio; fp.sharpen = sharpen; fp.scutoff = scutoff; fp.svr = svr;
    fp.smin = smin; fp.smax = smax;
    fp.interlaced = interlaced; fp.wintype = wintype;
    fp.pcutoff = pcutoff; fp.pfactor = pfactor;
    fp.pframe = pframe; fp.px = px; fp.py = py;
    fp.sigma2 = sigma2; fp.sigma3 = sigma3; fp.sigma4 = sigma4;
    fp.degrid = degrid; fp.dehalo = dehalo; fp.hr = hr; fp.ht = ht;
    fp.l = l; fp.t = t; fp.r = r; fp.b = b;
    fp.bits = bits; fp.bpp = bpp;
    fp.planeW = vi->width; fp.planeH = vi->height;
    fp.xShift = chroma ? vi->format.subSamplingW : 0;
    fp.yShift = chroma ? vi->format.subSamplingH : 0;
    fp.isChroma = chroma ? 1 : 0;

    std::string perr;
    auto pipe = gpu::GpuPipeline::acquire(fp, plane, FFT3D_CL_SRC, device_id, platform_id, perr);
    if (!pipe) {
      fail(("fft3dneocl: " + perr).c_str());
      return;
    }
    data->cl[plane] = chroma ? (l >> vi->format.subSamplingW) : l;
    data->ct[plane] = chroma ? (t >> vi->format.subSamplingH) : t;
    data->pipes[plane] = std::move(pipe);
  }

  VSFilterDependency deps[] = {{clip, rpStrictSpatial}};
  // original VSMode used fmSerial for bt=0 (Kalman state is sequential). API4
  // removed fmSerial; fmUnordered is its replacement: the core serializes all
  // requests to this instance (frames must still be consumed in order, which
  // is exactly what the original fmSerial filter required of its host).
  VSNode *node = vsapi->createVideoFilter2(
      "FFT3D", vi, fft3dGetFrame, fft3dFree,
      bt == 0 ? fmUnordered : fmParallelRequests, deps, 1,
      data.release(), core);
  if (!node) { fail("fft3dneocl: node creation failed"); return; }
  gpu::log_msg("instance create");
  vsapi->mapSetNode(out, "clip", node, maReplace);
  vsapi->freeNode(node);
}

void VS_CC fft3dFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
  gpu::log_msg("instance free");
  FFT3DData *d = reinterpret_cast<FFT3DData*>(instanceData);
  // API4 ownership: createVideoFilter2 does NOT consume the dependency
  // reference we got from mapGetNode in fft3dCreate — the filter owns it and
  // must return it here (official sdk/filter_skeleton.c; same fix as fft3dcl
  // commit bf54ad3). Without this the source node and its decoded-frame cache
  // are never destroyed: every StaxRip preview retained its full working set
  // until the process exited.
  if (d->child)
    vsapi->freeNode(d->child);
  delete d;
}

const VSFrame *VS_CC fft3dGetFrame(int n, int activationReason, void *instanceData,
                                   void **, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
  FFT3DData *d = reinterpret_cast<FFT3DData*>(instanceData);
  const int nframes = d->vi.numFrames;
  // engine lines 647-650: frames inside the temporal radius use the 2D filter
  const int btcur = gpu::GpuPipeline::btcur_for(n, nframes, d->bt);
  const int from = btcur > 1 ? -(btcur / 2) : 0;
  const int to = btcur > 1 ? (btcur - 1) / 2 : 0;
  const int nsrc = to - from + 1;
  const int cur = nsrc / 2; // window index of frame n
  auto clampf = [&](int f) { return f < 0 ? 0 : (f >= nframes ? nframes - 1 : f); };

  if (activationReason == arInitial) {
    for (int i = from; i <= to; i++)
      vsapi->requestFrameFilter(clampf(n + i), d->child, frameCtx);
    if (d->needPatFrame)
      vsapi->requestFrameFilter(clampf(d->pframe), d->child, frameCtx);
    return nullptr;
  }
  if (activationReason != arAllFramesReady) return nullptr;

  std::vector<const VSFrame*> frames(nsrc, nullptr);
  for (int k = 0; k < nsrc; k++) {
    frames[k] = vsapi->getFrameFilter(clampf(n + from + k), d->child, frameCtx);
    if (!frames[k]) {
      for (auto f : frames) if (f) vsapi->freeFrame(f);
      return nullptr;
    }
  }
  const VSFrame *patFrame = nullptr;
  if (d->needPatFrame) {
    patFrame = vsapi->getFrameFilter(clampf(d->pframe), d->child, frameCtx);
    if (!patFrame) {
      for (auto f : frames) if (f) vsapi->freeFrame(f);
      return nullptr;
    }
  }

  const VSVideoInfo &vi = d->vi;
  VSFrame *dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, frames[cur], core);

  // crop-offset plane pointer of the pattern source frame (same handling as
  // the source frames below: the f2c kernel writes the cropped region)
  auto patPlane = [&](int plane, ptrdiff_t &stride) -> const uint8_t* {
    stride = vsapi->getStride(patFrame, plane);
    const uint8_t *sp = vsapi->getReadPtr(patFrame, plane);
    if (d->crop)
      sp += (size_t)d->ct[plane] * stride + (size_t)d->cl[plane] * vi.format.bytesPerSample;
    return sp;
  };

  // engine 789-791: bt=0 returns the first frame unchanged, the Kalman state
  // only starts at frame 1 (fmUnordered serializes the requests on this node).
  // Engine 543-575 runs before that early return: the pattern is estimated on
  // the very first GetFrame call, frame 0 included.
  if (d->bt == 0 && n == 0) {
    if (d->needPatFrame) {
      std::lock_guard<std::mutex> lk(d->mtx);
      for (int plane = 0; plane < vi.format.numPlanes; plane++) {
        gpu::GpuPipeline *pipe = d->pipes[plane].get();
        if (!pipe || !pipe->patternPending()) continue;
        ptrdiff_t ps;
        const uint8_t *pp = patPlane(plane, ps); // separate call: ps must be set first
        std::string err;
        if (!pipe->estimatePattern(pp, ps, err)) {
          vsapi->setFilterError(("fft3dneocl: " + err).c_str(), frameCtx);
          for (auto f : frames) vsapi->freeFrame(f);
          vsapi->freeFrame(patFrame);
          vsapi->freeFrame(dst);
          return nullptr;
        }
      }
    }
    for (int plane = 0; plane < vi.format.numPlanes; plane++) {
      int pw = plane == 0 ? vi.width : vi.width >> vi.format.subSamplingW;
      int ph = plane == 0 ? vi.height : vi.height >> vi.format.subSamplingH;
      vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane),
                  vsapi->getReadPtr(frames[cur], plane), vsapi->getStride(frames[cur], plane),
                  (size_t)pw * vi.format.bytesPerSample, ph);
    }
    for (auto f : frames) vsapi->freeFrame(f);
    if (patFrame) vsapi->freeFrame(patFrame);
    return dst;
  }

  std::vector<const uint8_t*> srcs(nsrc, nullptr);
  bool failed = false;
  {
    // the GPU buffers are shared by all requests on this node
    std::lock_guard<std::mutex> lk(d->mtx);
    for (int plane = 0; plane < vi.format.numPlanes && !failed; plane++) {
      int pw = plane == 0 ? vi.width : vi.width >> vi.format.subSamplingW;
      int ph = plane == 0 ? vi.height : vi.height >> vi.format.subSamplingH;
      ptrdiff_t ss = vsapi->getStride(frames[cur], plane);
      ptrdiff_t ds = vsapi->getStride(dst, plane);
      const uint8_t *sp = vsapi->getReadPtr(frames[cur], plane);
      uint8_t *dp = vsapi->getWritePtr(dst, plane);

      gpu::GpuPipeline *pipe = d->pipes[plane] ? d->pipes[plane].get() : nullptr;
      if (!pipe) {
        vsh::bitblt(dp, ds, sp, ss, (size_t)pw * vi.format.bytesPerSample, ph);
        continue;
      }
      // engine 543-575: estimate the pattern once, before the first filtered
      // frame of this plane
      if (d->needPatFrame && pipe->patternPending()) {
        ptrdiff_t ps;
        const uint8_t *pp = patPlane(plane, ps); // separate call: ps must be set first
        std::string perr;
        if (!pipe->estimatePattern(pp, ps, perr)) {
          vsapi->setFilterError(("fft3dneocl: " + perr).c_str(), frameCtx);
          failed = true;
          continue;
        }
      }
      for (int k = 0; k < nsrc; k++) srcs[k] = vsapi->getReadPtr(frames[k], plane);

      if (d->crop) {
        // margins come from the source, the engine only writes the cropped region
        vsh::bitblt(dp, ds, sp, ss, (size_t)pw * vi.format.bytesPerSample, ph);
        int cl = d->cl[plane], ct = d->ct[plane];
        size_t off = (size_t)ct * ss + (size_t)cl * vi.format.bytesPerSample;
        for (int k = 0; k < nsrc; k++) srcs[k] += off;
        std::string err;
        if (!pipe->process(srcs.data(), nsrc, n + from, ss, dp + off, ds, err)) {
          vsapi->setFilterError(("fft3dneocl: " + err).c_str(), frameCtx);
          failed = true;
        }
      } else {
        std::string err;
        if (!pipe->process(srcs.data(), nsrc, n + from, ss, dp, ds, err)) {
          vsapi->setFilterError(("fft3dneocl: " + err).c_str(), frameCtx);
          failed = true;
        }
      }
    }
  }

  for (auto f : frames) vsapi->freeFrame(f);
  if (patFrame) vsapi->freeFrame(patFrame);
  if (failed) {
    vsapi->freeFrame(dst);
    return nullptr;
  }
  return dst;
}

// ---- device listing helper (new function, documented addition) ----
void VS_CC listDevices(const VSMap *, VSMap *out, void *, VSCore *, const VSAPI *vsapi) {
  std::string s;
  auto& api = ocl::shared_api();
  char err[256];
  if (!api.load(err, sizeof(err))) {
    s = std::string("OpenCL runtime not available: ") + err;
  } else {
    auto devs = ocl::Context::list_devices(api, true);
    for (size_t i = 0; i < devs.size(); i++) {
      s += std::to_string(i) + ": " + devs[i].device_name + " [" + devs[i].platform_name + "] " + devs[i].version + "\n";
    }
    if (s.empty()) s = "no OpenCL devices found";
  }
  vsapi->mapSetData(out, "data", s.c_str(), (int)s.size(), dtUtf8, maReplace);
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
  vspapi->configPlugin("com.github.fft3dneocl", "fft3dneocl",
                       "FFT3D denoiser, OpenCL GPU port of neo_FFT3D",
                       VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
  vspapi->registerFunction(
      "FFT3D",
      "clip:vnode;"
      "sigma:float:opt;beta:float:opt;planes:int[]:opt;bw:int:opt;bh:int:opt;bt:int:opt;"
      "ow:int:opt;oh:int:opt;kratio:float:opt;sharpen:float:opt;scutoff:float:opt;svr:float:opt;"
      "smin:float:opt;smax:float:opt;measure:int:opt;interlaced:int:opt;wintype:int:opt;"
      "pframe:int:opt;px:int:opt;py:int:opt;pshow:int:opt;pcutoff:float:opt;pfactor:float:opt;"
      "sigma2:float:opt;sigma3:float:opt;sigma4:float:opt;degrid:float:opt;dehalo:float:opt;"
      "hr:float:opt;ht:float:opt;l:int:opt;t:int:opt;r:int:opt;b:int:opt;"
      "opt:int:opt;ncpu:int:opt;mt:int:opt;"
      "device_id:int:opt;platform_id:int:opt;num_streams:int:opt",
      "clip:vnode", fft3dCreate, nullptr, plugin);
  vspapi->registerFunction("ListDevices", "", "data:data", listDevices, nullptr, plugin);
}
