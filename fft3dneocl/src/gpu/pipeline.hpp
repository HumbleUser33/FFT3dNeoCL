// GpuPipeline: the validated kernel chain (frame_to_cover -> ... ->
// cover_to_frame) for one plane. Geometry and tables fixed at create();
// process() runs one output frame of one plane, using nsrc source frames
// (nsrc == 1 for the 2D path, nsrc == bt for the temporal Wiener path).
#pragma once

#include <string>
#include <cstddef>
#include <memory>

#include "../ocl/ocl_loader.hpp"
#include "../ocl/context.hpp"
#include "tables.hpp"

namespace gpu {

// diagnostic log (FFT3DNEOCL_LOG=1): pool + instance lifetime tracing
void log_msg(const char* what);

class GpuPipeline {
public:
  GpuPipeline() = default;
  ~GpuPipeline();
  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // p: params with original defaults; sigma scaling is applied here.
  // kernel_src: the full fft3d.cl source (embedded in the plugin binary).
  bool create(const FilterParams& p, const char* kernel_src,
              int device_id, int platform_id, std::string& err);

  // Shared pipeline pool: identical filter parameters reuse the same pipeline
  // (device buffers + compiled program) across filter instances. StaxRip
  // recreates the whole filter graph on every preview open; churning ~4 GB of
  // device buffers per cycle is what ratchets the process DRAM usage. The pool
  // keeps pipelines alive while in use and prunes them when idle.
  // plane disambiguates U and V chroma (identical parameters otherwise).
  static std::shared_ptr<GpuPipeline> acquire(const FilterParams& p, int plane, const char* kernel_src,
                                              int device_id, int platform_id, std::string& err);

  // srcs[k] point to the (already crop-offset) plane of frame firstFrame+k
  // (firstFrame = n - btcur/2, so the current frame is at index btcur/2).
  // srcs[0] alone is used when nsrc == 1. Pitches in bytes.
  // Window frames whose raw spectrum is already in the GPU fftcache are not
  // re-transformed (same numbers, less work).
  bool process(const unsigned char* const* srcs, int nsrc, int firstFrame,
               ptrdiff_t src_pitch, unsigned char* dst, ptrdiff_t dst_pitch,
               std::string& err);

  // 2D fallback frame range of the original engine (fft3d_engine.h 647-650):
  // frames closer to the clip edges than the temporal radius use btcur = 1.
  static int btcur_for(int n, int numFrames, int bt);

  // pattern denoise (engine 543-575): one forward FFT of the pattern source
  // frame, then FindPatternBlock + SetPattern (helper.cpp) on the host with the
  // exact CPU float operation order; the resulting pattern2d goes to d_pat.
  // Called once per instance, before the first process().
  bool estimatePattern(const unsigned char* src, ptrdiff_t src_pitch, std::string& err);
  bool patternPending() const { return p.pfactor != 0.0f && !patternSet; }

  const Geometry& geom() const { return g; }

private:
  bool run_fft_fwd(cl_mem src, cl_mem dst, std::string& err);
  bool run_fft_inv(cl_mem spec, cl_mem real, std::string& err);
  bool run_gridfraction(cl_mem spec, cl_mem gf_out, std::string& err);
  bool ensure_frame(ptrdiff_t src_pitch, std::string& err);
  // host<->device transfers via map/unmap: the driver maps the buffer's own
  // backing instead of allocating a fresh pinned staging copy per transfer
  // (the per-transfer pinned copies are what ratchet the process DRAM usage
  // across StaxRip preview cycles). Same bytes, bit-exact.
  bool upload_frame(const unsigned char* src, size_t n, std::string& err);
  void free_buffers();
  // re-initialize per-instance temporal state when a pooled pipeline is handed
  // to a new filter instance (fftcache, pattern, Kalman state)
  bool reset_state(std::string& err);

  FilterParams p;
  Geometry g {};
  Tables t {};
  ocl::Context ctx;
  int fused = 1;
  int odist = 0;

  // kernels
  cl_kernel kfc = nullptr, kco = nullptr, koc = nullptr, kcf = nullptr;
  cl_kernel kgf = nullptr, kap = nullptr, k3d = nullptr, ksp = nullptr;
  cl_kernel kkal = nullptr;                            // kalman (bt=0)
  cl_kernel kfwd = nullptr, kinv = nullptr;             // fused
  cl_kernel kfr = nullptr, kfc2 = nullptr, kir = nullptr, kic = nullptr; // split

  // buffers
  cl_mem d_frame = nullptr, d_cover = nullptr, d_in = nullptr, d_spec = nullptr;
  cl_mem d_gf = nullptr, d_gf2 = nullptr, d_rinv = nullptr, d_cover2 = nullptr, d_dst = nullptr;
  cl_mem d_twR = nullptr, d_twC = nullptr, d_tmp = nullptr, d_gs = nullptr;
  cl_mem d_pat = nullptr, d_wsh = nullptr, d_wde = nullptr;
  cl_mem d_wanxl = nullptr, d_wanxr = nullptr, d_wanyl = nullptr, d_wanyr = nullptr;
  cl_mem d_wsynxl = nullptr, d_wsynxr = nullptr, d_wsynyl = nullptr, d_wsynyr = nullptr;
  // GPU fftcache: mirrors the CPU cache<fftwf_complex>(bt+3, outsize)
  // (fft3d_engine.h 158, 727-760): raw spectra keyed by frame number, LRU.
  static const int kCacheMax = 8; // bt + 3, bt <= 5
  cl_mem d_specC[kCacheMax] = {};
  int cFrame[kCacheMax] = {};      // frame number held by the slot, -1 = free
  unsigned cStamp[kCacheMax] = {}; // LRU stamp (0 = never used)
  unsigned cClock = 0;
  int cEntries = 0;                // active slots (bt + 3)
  cl_mem d_spec_out = nullptr; // apply3d output (the CPU writes to outrez)
  // Kalman (bt=0) persistent state, mirrors the engine's outLast/covar/
  // covarProcess (fft3d_engine.h 144-149, init 346-351): every frame n >= 1
  // reads and rewrites them, so they live for the whole pipeline instance.
  cl_mem d_outLast = nullptr, d_covar = nullptr, d_covarProcess = nullptr;

  // pattern denoise (pfactor != 0): host copies of the CPU tables and the
  // estimated pattern (helper.cpp FindPatternBlock/SetPattern run on the host,
  // in the exact CPU float order); patX/patY = selected block, psigma = mean
  // sigma (diagnostic only, pshow is not implemented).
  std::vector<float> h_pwin, h_pattern2d, h_gsblk; // h_gsblk: gridsample block 0
  bool patternSet = false;
  int patX = 0, patY = 0;
  float psigma = 0.0f;

  ptrdiff_t frame_pitch = 0, dst_pitch0 = 0; // staging buffers sized on first process()
  // read-back staging: reused across process() calls. A fresh 6 MB vector per
  // plane per frame churns the CRT heap and leaves it fragmented (the
  // StaxRip preview-open DRAM leak); one persistent buffer removes the churn.
  std::vector<unsigned char> h_stage;
};

} // namespace gpu
