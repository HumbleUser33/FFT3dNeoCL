#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace gpu {

GpuPipeline::~GpuPipeline() {
  if (!ctx.api.lib) { free_buffers(); return; }
  auto& api = ctx.api;
  cl_kernel ks[] = { kfc, kco, koc, kcf, kgf, kap, k3d, ksp, kkal, kfwd, kinv, kfr, kfc2, kir, kic };
  for (cl_kernel k : ks) if (k) api.ReleaseKernel(k);
  free_buffers();
}

// ---- shared pipeline pool ----
// Diagnostic log, enabled with FFT3DNEOCL_LOG=1 in the environment: traces
// pool reuse/creation and filter instance lifetime so a real-world leak
// (StaxRip preview) can be attributed precisely.
void log_msg(const char* what) {
  static const bool on = getenv("FFT3DNEOCL_LOG") != nullptr;
  if (!on) return;
  static std::mutex m;
  std::lock_guard<std::mutex> lk(m);
  static FILE* f = fopen((std::string(getenv("TEMP") ? getenv("TEMP") : ".") +
                          "/fft3dneocl_pool.log").c_str(), "a");
  if (!f) return;
  fprintf(f, "[%ld] pid=%u %s\n", (long)time(nullptr), (unsigned)_getpid(), what);
  fflush(f);
}

static std::string pipe_key(const gpu::FilterParams& p, int plane, int device_id, int platform_id) {
  char buf[640];
  snprintf(buf, sizeof(buf),
    "g%.9g|b%.9g|w%d|h%d|t%d|ow%d|oh%d|k%.9g|s%.9g|sc%.9g|svr%.9g|smin%.9g|smax%.9g"
    "|il%d|wt%d|pc%.9g|pf%.9g|s2%.9g|s3%.9g|s4%.9g|pfr%d|px%d|py%d|dg%.9g|dh%.9g|hr%.9g|ht%.9g"
    "|L%d|T%d|R%d|B%d|bi%d|bp%d|W%d|H%d|xs%d|ys%d|c%d|pn%d|dev%d|pl%d",
    p.sigma, p.beta, p.bw, p.bh, p.bt, p.ow, p.oh, p.kratio, p.sharpen, p.scutoff, p.svr,
    p.smin, p.smax, p.interlaced, p.wintype, p.pcutoff, p.pfactor, p.sigma2, p.sigma3, p.sigma4,
    p.pframe, p.px, p.py, p.degrid, p.dehalo, p.hr, p.ht, p.l, p.t, p.r, p.b,
    p.bits, p.bpp, p.planeW, p.planeH, p.xShift, p.yShift, p.isChroma, plane, device_id, platform_id);
  return buf;
}

// free-list pool shared by acquire() and the shared_ptr deleter
struct PipePool {
  std::mutex mtx;
  std::map<std::string, std::vector<GpuPipeline*>> idle; // owned raw pointers
  // VapourSynth unloads the plugin DLL (FreeLibrary) when a core is destroyed,
  // so a host that creates one core per preview (vspipe, and the leak harness)
  // would otherwise leak every idle pipeline's buffers at each unload. Static
  // destruction order: this namespace-scope pool is destroyed AFTER the
  // function-local api/context/queue pools, which is safe — the Api struct is
  // POD and OpenCL.dll stays mapped (we never FreeLibrary it), and OpenCL
  // refcounting keeps contexts/queues alive until our buffers are released.
  ~PipePool() {
    for (auto& kv : idle)
      for (GpuPipeline* q : kv.second)
        delete q;
    idle.clear();
  }
};
static PipePool g_pipe_pool;

std::shared_ptr<GpuPipeline> GpuPipeline::acquire(const FilterParams& p, int plane, const char* kernel_src,
                                                  int device_id, int platform_id, std::string& err) {
  // Free-list pool: every instance gets its OWN pipeline (no sharing races),
  // but a released pipeline returns to the pool instead of being destroyed.
  // StaxRip overlaps several script evaluations (main + preview); churning
  // multi-GB device buffer sets per evaluation is what drove the DRAM leak.
  // Idle pipelines are reused by the next evaluation; the pool is capped so
  // changing parameters repeatedly cannot grow without limit.
  std::string key = pipe_key(p, plane, device_id, platform_id);

  GpuPipeline* raw = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_pipe_pool.mtx);
    auto it = g_pipe_pool.idle.find(key);
    if (it != g_pipe_pool.idle.end() && !it->second.empty()) {
      raw = it->second.back();
      it->second.pop_back();
      log_msg(("acquire REUSE idle=" + std::to_string(it->second.size())).c_str());
    } else {
      log_msg(("acquire CREATE idle=" + std::to_string(it != g_pipe_pool.idle.end() ? it->second.size() : 0) +
               " key=" + key).c_str());
    }
  }
  const bool fresh = (raw == nullptr);
  if (fresh) raw = new GpuPipeline();
  if (fresh) {
    if (!raw->create(p, kernel_src, device_id, platform_id, err)) { delete raw; return nullptr; }
  } else if (!raw->reset_state(err)) {
    delete raw; return nullptr;
  }
  // custom deleter: hand the pipeline back to the pool instead of deleting
  std::shared_ptr<GpuPipeline> sp(raw, [key](GpuPipeline* q) {
    std::lock_guard<std::mutex> lk(g_pipe_pool.mtx);
    auto& v = g_pipe_pool.idle[key];
    size_t total = 0;
    for (auto& kv : g_pipe_pool.idle) total += kv.second.size();
    log_msg(("pool release: idle=" + std::to_string(v.size()) + " total=" + std::to_string(total) +
             " key=" + key).c_str());
    if (v.size() < 2 && total < 6) v.push_back(q); // keep idle for reuse
    else delete q;                                  // over cap: really destroy
  });
  return sp;
}

bool GpuPipeline::reset_state(std::string& err) {
  for (int i = 0; i < kCacheMax; i++) { cFrame[i] = -1; cStamp[i] = 0; }
  cClock = 0;
  patternSet = false; // pattern (pfactor) is re-estimated for the new instance
  if (p.bt == 0 && d_outLast) { // Kalman persistent state: re-init like create()
    auto& api = ctx.api;
    float zero = 0.0f, sig2d = t.sigmaSquaredNoiseNormed2D;
    size_t state_bytes = (size_t)g.outsize * 8;
    if (api.EnqueueFillBuffer(ctx.queue, d_outLast, &zero, 4, 0, state_bytes, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueFillBuffer(ctx.queue, d_covar, &sig2d, 4, 0, state_bytes, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueFillBuffer(ctx.queue, d_covarProcess, &sig2d, 4, 0, state_bytes, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "kalman state reset"; return false;
    }
  }
  return true;
}

void GpuPipeline::free_buffers() {
  if (!ctx.api.lib) return;
  auto& api = ctx.api;
  cl_mem bs[] = { d_frame, d_cover, d_in, d_spec, d_gf, d_gf2, d_rinv, d_cover2, d_dst,
                  d_twR, d_twC, d_tmp, d_gs, d_pat, d_wsh, d_wde, d_spec_out,
                  d_outLast, d_covar, d_covarProcess,
                  d_wanxl, d_wanxr, d_wanyl, d_wanyr,
                  d_wsynxl, d_wsynxr, d_wsynyl, d_wsynyr,
                  d_specC[0], d_specC[1], d_specC[2], d_specC[3],
                  d_specC[4], d_specC[5], d_specC[6], d_specC[7] };
  for (cl_mem b : bs) if (b) api.ReleaseMemObject(b);
  d_frame = d_cover = d_in = d_spec = d_gf = d_gf2 = d_rinv = d_cover2 = d_dst = nullptr;
  d_twR = d_twC = d_tmp = d_gs = d_pat = d_wsh = d_wde = d_spec_out = nullptr;
  d_outLast = d_covar = d_covarProcess = nullptr;
  d_wanxl = d_wanxr = d_wanyl = d_wanyr = nullptr;
  d_wsynxl = d_wsynxr = d_wsynyl = d_wsynyr = nullptr;
  for (int i = 0; i < kCacheMax; i++) d_specC[i] = nullptr;
  std::vector<unsigned char>().swap(h_stage); // return the host staging buffer too
}

// engine lines 647-650: 2D Wiener for the frames inside the temporal radius
int GpuPipeline::btcur_for(int n, int numFrames, int bt) {
  if (bt < 2) return bt > 0 ? 1 : bt;
  if (bt / 2 > n) return 1;
  if ((bt - 1) / 2 > (numFrames - 1 - n)) return 1;
  return bt;
}

bool GpuPipeline::create(const FilterParams& in_p, const char* kernel_src,
                         int device_id, int platform_id, std::string& err) {
  p = in_p;
  scale_sigmas(p); // engine ctor lines 98-110 (sigma/smin/smax by bit-depth factor)
  if (!compute_geometry(p, g)) { err = "invalid ow/oh or frame too small for blocks"; return false; }
  make_tables(p, g, t);
  odist = g.outpitch * p.bh;

  // engine 373-378: different sigma2/3/4 with pfactor == 0 build the pattern
  // from the sigmas and force pfactor = 1 (no pattern frame is fetched).
  if ((p.sigma2 != p.sigma || p.sigma3 != p.sigma || p.sigma4 != p.sigma) && p.pfactor == 0.0f) {
    h_pattern2d = sigmas_to_pattern(p, g, t.norm);
    patternSet = true;
    p.pfactor = 1.0f;
  }
  h_pwin = make_pwin(p, g); // engine ctor 353-368

  // fused FFT if the block fits in device local memory (same rule as fft_test).
  // Probe devices for a first guess; ctx.create may pick another device, so the
  // choice is re-checked after create and the program rebuilt if needed.
  int fused_guess = 1;
  {
    auto& api = ocl::shared_api();
    char e2[256];
    if (api.load(e2, sizeof(e2))) {
      const auto& dv = ocl::cached_devices(api);
      cl_ulong need = (cl_ulong)2 * p.bh * (p.bw + 1) * 8;
      fused_guess = (dv.empty() || dv[0].local_mem >= need) ? 1 : 0;
    }
  }
  fused = fused_guess;
  char bopts[160];
  snprintf(bopts, sizeof(bopts),
           "-cl-std=CL1.2 -DBW=%d -DBH=%d -DOW=%d -DOH=%d -DBPP=%d -DFFT_FUSED=%d -DBT=%d",
           p.bw, p.bh, g.ow, g.oh, p.bpp, fused, p.bt);

  char cerr[4096];
  if (!ctx.create(kernel_src, bopts, device_id, platform_id, cerr, sizeof(cerr))) {
    err = std::string("fft3dneocl: ") + cerr;
    return false;
  }
  // if the chosen device differs from the probe guess, rebuild with the right path
  cl_ulong need = (cl_ulong)2 * p.bh * (p.bw + 1) * 8;
  if (fused && ctx.info.local_mem < need) {
    free_buffers();
    auto& api = ctx.api;
    cl_kernel ks[] = { kfc, kco, koc, kcf, kgf, kap, k3d, ksp, kkal, kfwd, kinv, kfr, kfc2, kir, kic };
    for (cl_kernel k : ks) if (k) api.ReleaseKernel(k);
    kfc = kco = koc = kcf = kgf = kap = k3d = ksp = kfwd = kinv = kfr = kfc2 = kir = kic = nullptr;
    if (ctx.program) api.ReleaseProgram(ctx.program);
    // ctx.ctx and ctx.queue stay: owned by the shared pools; create() re-acquires
    ctx.program = nullptr;
    fused = 0;
    snprintf(bopts, sizeof(bopts),
             "-cl-std=CL1.2 -DBW=%d -DBH=%d -DOW=%d -DOH=%d -DBPP=%d -DFFT_FUSED=0 -DBT=%d",
             p.bw, p.bh, g.ow, g.oh, p.bpp, p.bt);
    if (!ctx.create(kernel_src, bopts, device_id, platform_id, cerr, sizeof(cerr))) {
      err = std::string("fft3dneocl: ") + cerr;
      return false;
    }
  }

  cl_int e = CL_SUCCESS;
  auto& api = ctx.api;
  auto K = [&](cl_kernel& k, const char* name) {
    k = ctx.kernel(name, &e);
    if (e != CL_SUCCESS) { err = std::string("kernel ") + name + ": " + ocl::err_str(e); return false; }
    return true;
  };
  if (!K(kfc, "frame_to_cover") || !K(kco, "cover_to_overlap") ||
      !K(koc, "overlap_to_cover") || !K(kcf, "cover_to_frame") ||
      !K(kgf, "gridfraction_k") || !K(kap, "apply2d")) return false;
  if (p.bt > 1 && !K(k3d, "apply3d")) return false;
  if ((p.bt > 1 || p.bt == -1 || p.bt == 0 || (p.bt == 1 && p.pfactor != 0.0f)) && !K(ksp, "sharpen_post")) return false;
  if (p.bt == 0 && !K(kkal, "kalman")) return false;
  if (fused) {
    if (!K(kfwd, "fft_fwd") || !K(kinv, "fft_inv")) return false;
  } else {
    if (!K(kfr, "fft_fwd_rows") || !K(kfc2, "fft_fwd_cols") ||
        !K(kir, "fft_inv_rows") || !K(kic, "fft_inv_cols")) return false;
  }

  size_t device_bytes = 0; // VRAM footprint report (mirrors fft3dcl's log line)
  auto Buf = [&](cl_mem& b, cl_mem_flags f, size_t sz, const void* host) {
    b = api.CreateBuffer(ctx.ctx, f, sz, (void*)host, &e);
    if (e != CL_SUCCESS) { err = "CreateBuffer: " + ocl::err_str(e); return false; }
    device_bytes += sz;
    return true;
  };
  const size_t cover_bytes = (size_t)g.coverheight * g.coverpitch * p.bpp;
  if (!Buf(d_twR, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (2 * (p.bw - 1)) * 4, twiddle_table(p.bw).data())) return false;
  // twiddle_table returns a temporary; keep host copies alive for COPY_HOST_PTR:
  // (CreateBuffer copies synchronously, so temporaries are fine)
  if (!Buf(d_twC, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (2 * (p.bh - 1)) * 4, twiddle_table(p.bh).data())) return false;
  if (!Buf(d_wanxl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.ow * 4, t.wanxl.data())) return false;
  if (!Buf(d_wanxr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.ow * 4, t.wanxr.data())) return false;
  if (!Buf(d_wanyl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.oh * 4, t.wanyl.data())) return false;
  if (!Buf(d_wanyr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.oh * 4, t.wanyr.data())) return false;
  if (!Buf(d_wsynxl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.ow * 4, t.wsynxl.data())) return false;
  if (!Buf(d_wsynxr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.ow * 4, t.wsynxr.data())) return false;
  if (!Buf(d_wsynyl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.oh * 4, t.wsynyl.data())) return false;
  if (!Buf(d_wsynyr, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, g.oh * 4, t.wsynyr.data())) return false;
  if (!Buf(d_wsh, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)p.bh * g.outpitch * 4, t.wsharpen.data())) return false;
  if (!Buf(d_wde, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)p.bh * g.outpitch * 4, t.wdehalo.data())) return false;
  {
    // pattern2d buffer: zero until estimated; pre-filled for the sigmas variant
    std::vector<float> pat((size_t)p.bh * g.outpitch, 0.0f);
    if (patternSet) pat = h_pattern2d;
    if (!Buf(d_pat, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pat.size() * 4, pat.data())) return false;
  }
  if (!Buf(d_cover, CL_MEM_READ_WRITE, cover_bytes, nullptr)) return false;
  if (!Buf(d_cover2, CL_MEM_READ_WRITE, cover_bytes, nullptr)) return false;
  if (!Buf(d_in, CL_MEM_READ_WRITE, (size_t)g.insize * 4, nullptr)) return false;
  if (!Buf(d_rinv, CL_MEM_READ_WRITE, (size_t)g.insize * 4, nullptr)) return false;
  if (!Buf(d_spec, CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
  if (!Buf(d_gs, CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
  if (!Buf(d_gf, CL_MEM_READ_WRITE, (size_t)g.nblocks * 4, nullptr)) return false;
  if (p.bt > 1) {
    // fftcache slots: one spectrum per frame of the temporal window + 3 spare
    // (CPU: cache(bt+3, outsize)), plus the apply3d output
    cEntries = p.bt + 3;
    for (int i = 0; i < cEntries; i++) {
      if (!Buf(d_specC[i], CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
      cFrame[i] = -1; cStamp[i] = 0;
    }
    if (!Buf(d_spec_out, CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
    if (!Buf(d_gf2, CL_MEM_READ_WRITE, (size_t)g.nblocks * 4, nullptr)) return false;
  }
  if (!fused && !Buf(d_tmp, CL_MEM_READ_WRITE, (size_t)g.insize * 8, nullptr)) return false;
  if (p.bt == 0) {
    // persistent Kalman state (engine 144-149), initialized like engine 346-351:
    // outLast = 0, covar = covarProcess = sigmaSquaredNoiseNormed2D (both parts)
    if (!Buf(d_outLast, CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
    if (!Buf(d_covar, CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
    if (!Buf(d_covarProcess, CL_MEM_READ_WRITE, (size_t)g.outsize * 8, nullptr)) return false;
    float zero = 0.0f, sig2d = t.sigmaSquaredNoiseNormed2D;
    size_t state_bytes = (size_t)g.outsize * 8;
    if (api.EnqueueFillBuffer(ctx.queue, d_outLast, &zero, 4, 0, state_bytes, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueFillBuffer(ctx.queue, d_covar, &sig2d, 4, 0, state_bytes, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueFillBuffer(ctx.queue, d_covarProcess, &sig2d, 4, 0, state_bytes, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "kalman state fill"; return false;
    }
  }
  fprintf(stderr, "fft3dneocl: %.1f MiB of device buffers\n", device_bytes / 1048576.0);
  log_msg(("pipeline create: " + std::to_string(device_bytes / 1048576) + " MiB buffers").c_str());

  // constant kernel args (per-plane geometry and tables)
  auto A = [&](cl_kernel k, int i, const void* v) {
    if (api.SetKernelArg(k, i, sizeof(cl_mem), v) != CL_SUCCESS) { err = "SetKernelArg mem"; return false; }
    return true;
  };
  auto AI = [&](cl_kernel k, int i, int v) {
    if (api.SetKernelArg(k, i, sizeof(int), &v) != CL_SUCCESS) { err = "SetKernelArg int"; return false; }
    return true;
  };
  auto AF = [&](cl_kernel k, int i, float v) {
    if (api.SetKernelArg(k, i, sizeof(float), &v) != CL_SUCCESS) { err = "SetKernelArg float"; return false; }
    return true;
  };
  // frame_to_cover: 0 src, 1 src_pitch set in process()
  if (!AI(kfc, 2, g.frameW) || !AI(kfc, 3, g.frameH) || !A(kfc, 4, &d_cover) ||
      !AI(kfc, 5, g.coverpitch) || !AI(kfc, 6, g.coverwidth) || !AI(kfc, 7, g.coverheight) ||
      !AI(kfc, 8, g.mirw) || !AI(kfc, 9, g.mirh) || !AI(kfc, 10, p.interlaced)) return false;
  // cover_to_overlap
  if (!A(kco, 0, &d_cover) || !AI(kco, 1, g.coverpitch) || !A(kco, 2, &d_in) ||
      !A(kco, 3, &d_wanxl) || !A(kco, 4, &d_wanxr) || !A(kco, 5, &d_wanyl) || !A(kco, 6, &d_wanyr) ||
      !AI(kco, 7, g.coverwidth) || !AI(kco, 8, g.coverheight) || !AI(kco, 9, g.nox) || !AI(kco, 10, g.noy) ||
      !AF(kco, 11, g.planeBase)) return false;
  // overlap_to_cover
  if (!A(koc, 0, &d_rinv) || !A(koc, 1, &d_cover2) || !AI(koc, 2, g.coverpitch) ||
      !A(koc, 3, &d_wsynxl) || !A(koc, 4, &d_wsynxr) || !A(koc, 5, &d_wsynyl) || !A(koc, 6, &d_wsynyr) ||
      !AI(koc, 7, g.coverwidth) || !AI(koc, 8, g.coverheight) || !AI(koc, 9, g.nox) || !AI(koc, 10, g.noy) ||
      !AF(koc, 11, t.norm) || !AF(koc, 12, g.rounder) || !AF(koc, 13, g.planeBase) || !AF(koc, 14, g.maxval)) return false;
  // cover_to_frame: 2 dst, 3 dst_pitch set in process()
  if (!A(kcf, 0, &d_cover2) || !AI(kcf, 1, g.coverpitch) ||
      !AI(kcf, 4, g.frameW) || !AI(kcf, 5, g.frameH) ||
      !AI(kcf, 6, g.mirw) || !AI(kcf, 7, g.mirh) || !AI(kcf, 8, p.interlaced)) return false;
  // gridfraction_k
  if (!A(kgf, 0, &d_spec) || !A(kgf, 1, &d_gs) || !AF(kgf, 2, p.degrid) ||
      !AI(kgf, 3, odist) || !A(kgf, 4, &d_gf)) return false;
  // apply2d
  if (!A(kap, 0, &d_spec) || !A(kap, 1, &d_gf) || !A(kap, 2, &d_pat) || !A(kap, 3, &d_gs) ||
      !A(kap, 4, &d_wsh) || !A(kap, 5, &d_wde) ||
      !AI(kap, 6, g.outpitch) || !AI(kap, 7, g.outwidth) || !AI(kap, 8, p.bh) ||
      !AF(kap, 9, t.sigmaSquaredNoiseNormed2D) || !AF(kap, 10, t.lowlimit) ||
      !AF(kap, 11, p.sharpen) || !AF(kap, 12, p.dehalo) ||
      !AF(kap, 13, t.sigmaSquaredSharpenMinNormed) || !AF(kap, 14, t.sigmaSquaredSharpenMaxNormed) ||
      !AF(kap, 15, t.ht2n) || !AI(kap, 16, p.pfactor != 0.0f ? 1 : 0)) return false;
  if (p.bt > 1) {
    // apply3d buffer args 0..4 (the window spectra) are set per frame from the
    // fftcache; the rest is constant
    if (!A(k3d, 5, &d_spec_out) || !A(k3d, 6, &d_gf) || !A(k3d, 7, &d_gs) ||
        !AI(k3d, 8, g.outpitch) || !AI(k3d, 9, g.outwidth) || !AI(k3d, 10, p.bh) ||
        !AF(k3d, 12, t.lowlimit) || !A(k3d, 13, &d_pat) ||
        !AI(k3d, 14, p.pfactor != 0.0f ? 1 : 0)) return false;
    if (!A(ksp, 0, &d_spec_out) || !A(ksp, 1, &d_gf2) || !A(ksp, 2, &d_gs) ||
        !A(ksp, 3, &d_wsh) || !A(ksp, 4, &d_wde) ||
        !AI(ksp, 5, g.outpitch) || !AI(ksp, 6, g.outwidth) || !AI(ksp, 7, p.bh) ||
        !AF(ksp, 8, p.sharpen) || !AF(ksp, 9, p.dehalo) ||
        !AF(ksp, 10, t.sigmaSquaredSharpenMinNormed) ||
        !AF(ksp, 11, t.sigmaSquaredSharpenMaxNormed) || !AF(ksp, 12, t.ht2n)) return false;
  }
  if (p.bt == -1 || p.bt == 0 || (p.bt == 1 && p.pfactor != 0.0f)) {
    // sharpen only / kalman / (2D+pattern): Sharpen runs in-place on the
    // spectrum that reaches fft_inv (raw spectrum for bt=-1, outLast copy for
    // bt=0; engine 836-865, 817), its gridfraction reads that same spectrum's DC
    if (!A(ksp, 0, &d_spec) || !A(ksp, 1, &d_gf) || !A(ksp, 2, &d_gs) ||
        !A(ksp, 3, &d_wsh) || !A(ksp, 4, &d_wde) ||
        !AI(ksp, 5, g.outpitch) || !AI(ksp, 6, g.outwidth) || !AI(ksp, 7, p.bh) ||
        !AF(ksp, 8, p.sharpen) || !AF(ksp, 9, p.dehalo) ||
        !AF(ksp, 10, t.sigmaSquaredSharpenMinNormed) ||
        !AF(ksp, 11, t.sigmaSquaredSharpenMaxNormed) || !AF(ksp, 12, t.ht2n)) return false;
  }
  if (p.bt == 0) {
    // sfp.kratio2 = kratio * kratio (fft3d_engine.h 674), float product
    float kratio2 = p.kratio * p.kratio;
    if (!A(kkal, 0, &d_outLast) || !A(kkal, 1, &d_covar) || !A(kkal, 2, &d_covarProcess) ||
        !A(kkal, 3, &d_spec) ||
        !AI(kkal, 4, g.outpitch) || !AI(kkal, 5, p.bh) ||
        !AF(kkal, 6, t.sigmaSquaredNoiseNormed2D) || !AF(kkal, 7, kratio2) ||
        !A(kkal, 8, &d_pat) || !AI(kkal, 9, p.pfactor != 0.0f ? 1 : 0)) return false;
  }
  // fft constant args
  if (fused) {
    if (!A(kfwd, 2, &d_twR) || !A(kfwd, 3, &d_twC) || !AI(kfwd, 4, g.outpitch)) return false;
    if (!A(kinv, 2, &d_twR) || !A(kinv, 3, &d_twC) || !AI(kinv, 4, g.outpitch)) return false;
  } else {
    if (!A(kfr, 2, &d_twR)) return false;
    if (!A(kfc2, 2, &d_twC) || !AI(kfc2, 3, g.outpitch)) return false;
    if (!A(kir, 2, &d_twR) || !AI(kir, 3, g.outpitch)) return false;
    if (!A(kic, 2, &d_twC)) return false;
  }

  // gridsample: engine fills the cover with maxval, runs CoverToOverlap + one
  // forward FFT, and keeps the spectrum of block 0 (we compute all blocks).
  {
    std::vector<unsigned char> fill(cover_bytes, 0);
    size_t npx = (size_t)g.coverheight * g.coverpitch;
    if (p.bpp == 1) {
      std::memset(fill.data(), (int)g.maxval, cover_bytes);
    } else if (p.bpp == 2) {
      uint16_t v = (uint16_t)(int)g.maxval;
      for (size_t i = 0; i < npx; i++) std::memcpy(&fill[i * 2], &v, 2);
    } else {
      float v = 1.0f;
      for (size_t i = 0; i < npx; i++) std::memcpy(&fill[i * 4], &v, 4);
    }
    if (api.EnqueueWriteBuffer(ctx.queue, d_cover, CL_TRUE, 0, cover_bytes, fill.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "gridsample write"; return false;
    }
    size_t g2[2] = { (size_t)g.coverwidth, (size_t)g.coverheight };
    // engine line 408: CoverToOverlap(..., false) — no chroma centering even on
    // chroma planes, so force planeBase to 0 for this pass and restore after
    float zeroBase = 0.0f;
    if (api.SetKernelArg(kco, 11, sizeof(float), &zeroBase) != CL_SUCCESS) {
      err = "gridsample planeBase"; return false;
    }
    if (api.EnqueueNDRangeKernel(ctx.queue, kco, 2, nullptr, g2, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "gridsample cover_to_overlap"; return false;
    }
    if (!run_fft_fwd(d_in, d_gs, err)) return false;
    if (api.Finish(ctx.queue) != CL_SUCCESS) { err = "gridsample finish"; return false; }
    if (api.SetKernelArg(kco, 11, sizeof(float), &g.planeBase) != CL_SUCCESS) {
      err = "gridsample planeBase restore"; return false;
    }
    // host copy of gridsample block 0: FindPatternBlock/SetPattern run on the
    // host and only read gridsample[h*outpitch + w], h<bh, w<outwidth
    h_gsblk.resize((size_t)p.bh * g.outpitch * 2);
    if (api.EnqueueReadBuffer(ctx.queue, d_gs, CL_TRUE, 0, h_gsblk.size() * 4,
                              h_gsblk.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "gridsample readback"; return false;
    }
  }
  return true;
}

bool GpuPipeline::run_fft_fwd(cl_mem src, cl_mem dst, std::string& err) {
  auto& api = ctx.api;
  if (fused) {
    if (api.SetKernelArg(kfwd, 0, sizeof(cl_mem), &src) != CL_SUCCESS ||
        api.SetKernelArg(kfwd, 1, sizeof(cl_mem), &dst) != CL_SUCCESS) { err = "fft_fwd args"; return false; }
    size_t g1[1] = { (size_t)g.nblocks * 256 }, l1[1] = { 256 };
    if (api.EnqueueNDRangeKernel(ctx.queue, kfwd, 1, nullptr, g1, l1, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "fft_fwd enqueue"; return false;
    }
  } else {
    if (api.SetKernelArg(kfr, 0, sizeof(cl_mem), &src) != CL_SUCCESS ||
        api.SetKernelArg(kfr, 1, sizeof(cl_mem), &d_tmp) != CL_SUCCESS ||
        api.SetKernelArg(kfc2, 0, sizeof(cl_mem), &d_tmp) != CL_SUCCESS ||
        api.SetKernelArg(kfc2, 1, sizeof(cl_mem), &dst) != CL_SUCCESS) { err = "fft_fwd_rows/cols args"; return false; }
    size_t g1[1] = { (size_t)g.nblocks * p.bh * p.bw }, l1[1] = { (size_t)p.bw };
    size_t g2[1] = { (size_t)g.nblocks * (p.bw / 2 + 1) * p.bh }, l2[1] = { (size_t)p.bh };
    if (api.EnqueueNDRangeKernel(ctx.queue, kfr, 1, nullptr, g1, l1, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueNDRangeKernel(ctx.queue, kfc2, 1, nullptr, g2, l2, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "fft split fwd enqueue"; return false;
    }
  }
  return true;
}

bool GpuPipeline::run_fft_inv(cl_mem spec, cl_mem real, std::string& err) {
  auto& api = ctx.api;
  if (fused) {
    if (api.SetKernelArg(kinv, 0, sizeof(cl_mem), &spec) != CL_SUCCESS ||
        api.SetKernelArg(kinv, 1, sizeof(cl_mem), &real) != CL_SUCCESS) { err = "fft_inv args"; return false; }
    size_t g1[1] = { (size_t)g.nblocks * 256 }, l1[1] = { 256 };
    if (api.EnqueueNDRangeKernel(ctx.queue, kinv, 1, nullptr, g1, l1, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "fft_inv enqueue"; return false;
    }
  } else {
    if (api.SetKernelArg(kir, 0, sizeof(cl_mem), &spec) != CL_SUCCESS ||
        api.SetKernelArg(kir, 1, sizeof(cl_mem), &d_tmp) != CL_SUCCESS ||
        api.SetKernelArg(kic, 0, sizeof(cl_mem), &d_tmp) != CL_SUCCESS ||
        api.SetKernelArg(kic, 1, sizeof(cl_mem), &real) != CL_SUCCESS) { err = "fft_inv_rows/cols args"; return false; }
    size_t g1[1] = { (size_t)g.nblocks * p.bh * p.bw }, l1[1] = { (size_t)p.bw };
    size_t g2[1] = { (size_t)g.nblocks * p.bw * p.bh }, l2[1] = { (size_t)p.bh };
    if (api.EnqueueNDRangeKernel(ctx.queue, kir, 1, nullptr, g1, l1, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueNDRangeKernel(ctx.queue, kic, 1, nullptr, g2, l2, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "fft split inv enqueue"; return false;
    }
  }
  return true;
}

bool GpuPipeline::run_gridfraction(cl_mem spec, cl_mem gf_out, std::string& err) {
  auto& api = ctx.api;
  if (api.SetKernelArg(kgf, 0, sizeof(cl_mem), &spec) != CL_SUCCESS ||
      api.SetKernelArg(kgf, 4, sizeof(cl_mem), &gf_out) != CL_SUCCESS) { err = "gridfraction args"; return false; }
  size_t g1[1] = { (size_t)g.nblocks };
  if (api.EnqueueNDRangeKernel(ctx.queue, kgf, 1, nullptr, g1, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "gridfraction enqueue"; return false;
  }
  return true;
}

bool GpuPipeline::ensure_frame(ptrdiff_t src_pitch, std::string& err) {
  auto& api = ctx.api;
  if (!d_frame || frame_pitch != src_pitch) {
    if (d_frame) { api.ReleaseMemObject(d_frame); d_frame = nullptr; }
    cl_int e = CL_SUCCESS;
    size_t frame_bytes = (size_t)g.frameH * src_pitch;
    // host-backed buffer: map() returns a direct pointer into memory the
    // driver allocated ONCE, so per-frame transfers no longer stage extra
    // host memory that fragments the process heap against VapourSynth frames
    // (root cause of the DRAM ratchet at 4K). Fall back to device memory if
    // the driver rejects ALLOC_HOST_PTR.
    d_frame = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               frame_bytes, nullptr, &e);
    if (e != CL_SUCCESS)
      d_frame = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, frame_bytes, nullptr, &e);
    if (e != CL_SUCCESS) { err = "d_frame: " + ocl::err_str(e); return false; }
    int sp = (int)src_pitch; // kernel takes pitch in bytes as int
    if (api.SetKernelArg(kfc, 0, sizeof(cl_mem), &d_frame) != CL_SUCCESS ||
        api.SetKernelArg(kfc, 1, sizeof(int), &sp) != CL_SUCCESS) { err = "frame_to_cover args"; return false; }
    frame_pitch = src_pitch;
  }
  return true;
}

// transfer strategy switch for field diagnosis: FFT3DNEOCL_XFER=map (default)
// uses EnqueueMapBuffer; =copy uses EnqueueWriteBuffer/ReadBuffer through the
// persistent h_stage. The driver stages host memory differently for each.
static int xfer_mode() {
  static const int m = (getenv("FFT3DNEOCL_XFER") && strcmp(getenv("FFT3DNEOCL_XFER"), "map") == 0) ? 0 : 1;
  return m;
}

bool GpuPipeline::upload_frame(const unsigned char* src, size_t n, std::string& err) {
  auto& api = ctx.api;
  cl_int e;
  if (xfer_mode() == 1) { // copy: stage through our own persistent buffer so the
                          // driver never pins pages of the VapourSynth frame
    h_stage.resize(n);
    memcpy(h_stage.data(), src, n);
    if (api.EnqueueWriteBuffer(ctx.queue, d_frame, CL_TRUE, 0, n, h_stage.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "write frame"; return false;
    }
    return true;
  }
  // WRITE_INVALIDATE_REGION: we overwrite the whole region, skip the device->host sync
  void* p = api.EnqueueMapBuffer(ctx.queue, d_frame, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                 0, n, 0, nullptr, nullptr, &e);
  if (e != CL_SUCCESS) { err = "map frame: " + ocl::err_str(e); return false; }
  memcpy(p, src, n);
  if (api.EnqueueUnmapMemObject(ctx.queue, d_frame, p, 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "unmap frame"; return false;
  }
  return true;
}

// pattern estimation (engine 543-575): forward FFT of the pattern source frame,
// then FindPatternBlock + SetPattern (helper.cpp) on the host with the exact
// CPU float operation order, and upload the resulting pattern2d to d_pat.
bool GpuPipeline::estimatePattern(const unsigned char* src, ptrdiff_t src_pitch, std::string& err) {
  auto& api = ctx.api;
  if (!ensure_frame(src_pitch, err)) return false;
  size_t frame_bytes = (size_t)g.frameH * src_pitch;
  if (api.EnqueueWriteBuffer(ctx.queue, d_frame, CL_TRUE, 0, frame_bytes, src, 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "write pattern frame"; return false;
  }
  size_t gc[2] = { (size_t)g.coverwidth, (size_t)g.coverheight };
  if (api.EnqueueNDRangeKernel(ctx.queue, kfc, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS ||
      api.EnqueueNDRangeKernel(ctx.queue, kco, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "pattern stage enqueue"; return false;
  }
  if (!run_fft_fwd(d_in, d_spec, err)) return false;
  if (api.Finish(ctx.queue) != CL_SUCCESS) { err = "pattern fft finish"; return false; }

  std::vector<float> spec((size_t)g.outsize * 2);
  if (api.EnqueueReadBuffer(ctx.queue, d_spec, CL_TRUE, 0, spec.size() * 4,
                            spec.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "pattern readback"; return false;
  }

  // FindPatternBlock (helper.cpp 46-90): pick the block with the lowest
  // windowed grid-corrected power. Same scan order (by outer, bx inner) and
  // same sequential float accumulation as the CPU, so px/py match bit-exactly.
  if (p.px == 0 && p.py == 0) {
    float sigmaSquared = 1e15f;
    for (int by = 2; by < g.noy - 2; by++)
      for (int bx = 2; bx < g.nox - 2; bx++) {
        size_t base = ((size_t)g.nox * by + bx) * p.bh * g.outpitch;
        const float gcur = p.degrid * spec[base * 2] / h_gsblk[0]; // grid correction factor
        float acc = 0;
        for (int h = 0; h < p.bh; h++) {
          const float* row = &spec[(base + (size_t)h * g.outpitch) * 2];
          const float* grow = &h_gsblk[(size_t)h * g.outpitch * 2];
          const float* prow = h_pwin.data() + (size_t)h * g.outpitch;
          for (int w = 0; w < g.outwidth; w++) {
            float c0 = row[w * 2] - gcur * grow[w * 2];
            float c1 = row[w * 2 + 1] - gcur * grow[w * 2 + 1];
            float psd = c0 * c0 + c1 * c1;
            acc += psd * prow[w];
          }
        }
        if (acc < sigmaSquared) { patX = bx; patY = by; sigmaSquared = acc; }
      }
  } else {
    patX = p.px; patY = p.py; // fixed block requested by the user
    // the CPU reads out of range here (undefined behaviour); clamp instead so a
    // bad px/py cannot crash the plugin (valid inputs are unaffected)
    if (patX < 0) patX = 0; if (patX >= g.nox) patX = g.nox - 1;
    if (patY < 0) patY = 0; if (patY >= g.noy) patY = g.noy - 1;
  }

  // SetPattern (helper.cpp 92-131): pattern2d[h*outpitch+w] = psd*pwin[w] from
  // the selected block, grid-corrected; psigma is the mean sigma (diagnostic).
  h_pattern2d.assign((size_t)p.bh * g.outpitch, 0.0f);
  size_t base = ((size_t)g.nox * patY + patX) * p.bh * g.outpitch;
  float sigmaSquared = 0, weight = 0;
  for (int h = 0; h < p.bh; h++) {
    const float* prow = h_pwin.data() + (size_t)h * g.outpitch;
    for (int w = 0; w < g.outwidth; w++) weight += prow[w];
  }
  const float gcur = p.degrid * spec[base * 2] / h_gsblk[0];
  for (int h = 0; h < p.bh; h++) {
    const float* row = &spec[(base + (size_t)h * g.outpitch) * 2];
    const float* grow = &h_gsblk[(size_t)h * g.outpitch * 2];
    const float* prow = h_pwin.data() + (size_t)h * g.outpitch;
    float* drow = h_pattern2d.data() + (size_t)h * g.outpitch;
    for (int w = 0; w < g.outwidth; w++) {
      float c0 = row[w * 2] - gcur * grow[w * 2];
      float c1 = row[w * 2 + 1] - gcur * grow[w * 2 + 1];
      float psd = c0 * c0 + c1 * c1;
      drow[w] = psd * prow[w];
      sigmaSquared += drow[w];
    }
  }
  psigma = sqrtf(sigmaSquared / (weight * p.bh * g.outwidth));

  if (api.EnqueueWriteBuffer(ctx.queue, d_pat, CL_TRUE, 0, h_pattern2d.size() * 4,
                             h_pattern2d.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "pattern upload"; return false;
  }
  patternSet = true;
  return true;
}

bool GpuPipeline::process(const unsigned char* const* srcs, int nsrc, int firstFrame,
                          ptrdiff_t src_pitch, unsigned char* dst, ptrdiff_t dst_pitch,
                          std::string& err) {
  auto& api = ctx.api;
  size_t frame_bytes = (size_t)g.frameH * src_pitch;
  size_t dst_bytes = (size_t)g.frameH * dst_pitch;
  if (!ensure_frame(src_pitch, err)) return false;
  if (!d_dst || dst_pitch0 != dst_pitch) {
    if (d_dst) { api.ReleaseMemObject(d_dst); d_dst = nullptr; }
    cl_int e = CL_SUCCESS;
    d_dst = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dst_bytes, nullptr, &e);
    if (e != CL_SUCCESS)
      d_dst = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, dst_bytes, nullptr, &e);
    if (e != CL_SUCCESS) { err = "d_dst: " + ocl::err_str(e); return false; }
    int dp = (int)dst_pitch;
    if (api.SetKernelArg(kcf, 2, sizeof(cl_mem), &d_dst) != CL_SUCCESS ||
        api.SetKernelArg(kcf, 3, sizeof(int), &dp) != CL_SUCCESS) { err = "cover_to_frame args"; return false; }
    dst_pitch0 = dst_pitch;
  }

  size_t gc[2] = { (size_t)g.coverwidth, (size_t)g.coverheight };
  size_t gf[2] = { (size_t)g.frameW, (size_t)g.frameH };
  size_t ga[1] = { (size_t)g.nblocks * p.bh * g.outwidth };

  // forward pass: one per frame of the temporal window, unless its raw
  // spectrum is already in the fftcache (the CPU does the same with its
  // cache(bt+3, outsize), fft3d_engine.h 727-760)
  cl_mem win[5] = {};
  if (nsrc == 1) {
    if (!upload_frame(srcs[0], frame_bytes, err)) return false;
    if (api.EnqueueNDRangeKernel(ctx.queue, kfc, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS ||
        api.EnqueueNDRangeKernel(ctx.queue, kco, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "buffer stage enqueue"; return false;
    }
    if (!run_fft_fwd(d_in, d_spec, err)) return false;
  } else {
    for (int k = 0; k < nsrc; k++) {
      const int f = firstFrame + k;
      int slot = -1;
      for (int c = 0; c < cEntries; c++)
        if (cFrame[c] == f) { slot = c; break; }
      if (slot < 0) {
        // LRU victim, never a frame still needed by the current window
        unsigned best = 0;
        for (int c = 0; c < cEntries; c++) {
          if (cFrame[c] >= firstFrame && cFrame[c] < firstFrame + nsrc) continue;
          if (slot < 0 || cStamp[c] < best) { slot = c; best = cStamp[c]; }
        }
        if (slot < 0) slot = 0; // unreachable: cEntries = bt+3 > nsrc = bt
        if (!upload_frame(srcs[k], frame_bytes, err)) return false;
        if (api.EnqueueNDRangeKernel(ctx.queue, kfc, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS ||
            api.EnqueueNDRangeKernel(ctx.queue, kco, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
          err = "buffer stage enqueue"; return false;
        }
        if (!run_fft_fwd(d_in, d_specC[slot], err)) return false;
        cFrame[slot] = f;
      }
      cStamp[slot] = ++cClock;
      win[k] = d_specC[slot];
    }
  }

  cl_mem filtered = d_spec;
  if (nsrc == 1 && p.bt == 0) {
    // Kalman (engine 787-833): the kernel updates the persistent state and
    // writes the new output into d_outLast; the engine then memcpy's
    // outrez = outLast and runs Sharpen on that copy (only if set) before c2r.
    // Grid covers every padded column (bh*outpitch per block), like the CPU's
    // SIMD loop wrappers, so the mirror columns stay Hermitian.
    size_t gkal[1] = { (size_t)g.nblocks * p.bh * g.outpitch };
    if (api.EnqueueNDRangeKernel(ctx.queue, kkal, 1, nullptr, gkal, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "kalman enqueue"; return false;
    }
    if (api.EnqueueCopyBuffer(ctx.queue, d_outLast, d_spec, 0, 0, (size_t)g.outsize * 8, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "kalman copy"; return false;
    }
    if (p.sharpen != 0.0f || p.dehalo != 0.0f) {
      if (!run_gridfraction(d_spec, d_gf, err)) return false;
      if (api.EnqueueNDRangeKernel(ctx.queue, ksp, 1, nullptr, ga, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
        err = "sharpen_post enqueue"; return false;
      }
    }
  } else if (nsrc == 1 && p.bt == -1) {
    // sharpen only (engine 836-865): Sharpen runs on the raw spectrum and the
    // CPU returns early unless sharpen or dehalo is set (Sharpen.cpp 55)
    if (p.sharpen != 0.0f || p.dehalo != 0.0f) {
      if (!run_gridfraction(d_spec, d_gf, err)) return false;
      if (api.EnqueueNDRangeKernel(ctx.queue, ksp, 1, nullptr, ga, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
        err = "sharpen_post enqueue"; return false;
      }
    }
  } else if (nsrc == 1) {
    if (!run_gridfraction(d_spec, d_gf, err)) return false;
    if (api.EnqueueNDRangeKernel(ctx.queue, kap, 1, nullptr, ga, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "apply2d enqueue"; return false;
    }
    // engine 697-698: in 2D mode Sharpen runs only when pattern mode is on
    // (Apply2D skips the sharpen/dehalo factors for the pattern family);
    // Sharpen.cpp 55 returns early when both factors are 0
    if (p.pfactor != 0.0f && (p.sharpen != 0.0f || p.dehalo != 0.0f)) {
      if (!run_gridfraction(d_spec, d_gf, err)) return false;
      if (api.EnqueueNDRangeKernel(ctx.queue, ksp, 1, nullptr, ga, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
        err = "sharpen_post enqueue"; return false;
      }
    }
  } else {
    // CPU slot mapping (fft3d_engine.h 713-744): in[2+i] is the spectrum of
    // frame n+i; kernel arg sN is CPU slot N, window index k = N - 2 - from
    const int from = -(nsrc / 2);
    for (int slot = 0; slot < 5; slot++) {
      int k = slot - 2 - from;
      cl_mem b = win[(k >= 0 && k < nsrc) ? k : 0];
      if (api.SetKernelArg(k3d, slot, sizeof(cl_mem), &b) != CL_SUCCESS) { err = "apply3d spectra args"; return false; }
    }
    // gridfraction reads the DC of the CURRENT frame (CPU in[2][0][0])
    if (!run_gridfraction(win[nsrc / 2], d_gf, err)) return false;
    // engine line 657: btcur * sigma * sigma / norm, same multiplication order
    float sigma3d = ((float)nsrc * p.sigma) * p.sigma / t.norm;
    if (api.SetKernelArg(k3d, 11, sizeof(float), &sigma3d) != CL_SUCCESS) { err = "apply3d sigma arg"; return false; }
    if (api.EnqueueNDRangeKernel(ctx.queue, k3d, 1, nullptr, ga, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "apply3d enqueue"; return false;
    }
    filtered = d_spec_out;
    // the engine calls Sharpen after Apply3D, which returns early unless
    // sharpen or dehalo is set (code_impl/Sharpen.cpp line 55)
    if (p.sharpen != 0.0f || p.dehalo != 0.0f) {
      if (!run_gridfraction(d_spec_out, d_gf2, err)) return false;
      if (api.EnqueueNDRangeKernel(ctx.queue, ksp, 1, nullptr, ga, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
        err = "sharpen_post enqueue"; return false;
      }
    }
  }

  if (!run_fft_inv(filtered, d_rinv, err)) return false;
  if (api.EnqueueNDRangeKernel(ctx.queue, koc, 2, nullptr, gc, nullptr, 0, nullptr, nullptr) != CL_SUCCESS ||
      api.EnqueueNDRangeKernel(ctx.queue, kcf, 2, nullptr, gf, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
    err = "synthesis enqueue"; return false;
  }
  // read back through a staging buffer: d_dst rows contain only the processed
  // columns [0, frameW); copying full pitch would clobber the crop margins
  {
    cl_int e;
    size_t row_bytes = (size_t)g.frameW * p.bpp;
    if (xfer_mode() == 1) { // copy through the persistent host staging buffer
      h_stage.resize(dst_bytes);
      if (api.EnqueueReadBuffer(ctx.queue, d_dst, CL_TRUE, 0, dst_bytes, h_stage.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
        err = "read dst"; return false;
      }
      const unsigned char* sp = h_stage.data();
      for (int y = 0; y < g.frameH; y++)
        memcpy(dst + (size_t)y * dst_pitch, sp + (size_t)y * dst_pitch, row_bytes);
    } else {
    void* mp = api.EnqueueMapBuffer(ctx.queue, d_dst, CL_TRUE, CL_MAP_READ, 0, dst_bytes, 0, nullptr, nullptr, &e);
    if (e != CL_SUCCESS) { err = "map dst: " + ocl::err_str(e); return false; }
    const unsigned char* sp = (const unsigned char*)mp;
    for (int y = 0; y < g.frameH; y++)
      memcpy(dst + (size_t)y * dst_pitch, sp + (size_t)y * dst_pitch, row_bytes);
    if (api.EnqueueUnmapMemObject(ctx.queue, d_dst, mp, 0, nullptr, nullptr) != CL_SUCCESS) {
      err = "unmap dst"; return false;
    }
    }
  }
  if (api.Finish(ctx.queue) != CL_SUCCESS) { err = "finish"; return false; }
  return true;
}

} // namespace gpu
