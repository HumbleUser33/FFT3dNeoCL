// Host-side table generation, formulas copied verbatim from the original
// FFT3DEngine constructor (neo_FFT3D/src/fft3d_engine.h) so results match.
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

namespace gpu {

struct FilterParams {
  // user parameters (original names, original defaults)
  float sigma = 2.0f, beta = 1.0f;
  int bw = 32, bh = 32, bt = 3;
  int ow = -1, oh = -1;
  float kratio = 2.0f, sharpen = 0.0f, scutoff = 0.3f, svr = 1.0f;
  float smin = 4.0f, smax = 20.0f;
  int interlaced = 0, wintype = 0;
  float pcutoff = 0.1f, pfactor = 0.0f;
  float sigma2, sigma3, sigma4; // = sigma at init
  int pframe = 0, px = 0, py = 0; // pattern source frame / fixed block
  float degrid = 1.0f, dehalo = 0.0f, hr = 2.0f, ht = 50.0f;
  int l = 0, t = 0, r = 0, b = 0;

  // format (set by the plugin)
  int bits = 8, bpp = 1; // bpp: 1,2 (integer) or 4 (float32)
  int planeW = 0, planeH = 0; // FULL frame dimensions (not subsampled)
  int xShift = 0, yShift = 0; // 0 for luma/RGB, ssw/ssh for YUV chroma
  int isChroma = 0;
};

// geometry derived from params, same formulas as the engine
struct Geometry {
  int nox, noy;
  int ow, oh;          // resolved overlap (-1 -> bw/3, bh/3)
  int frameW, frameH;  // processed (cropped) frame dimensions
  int coverwidth, coverheight, coverpitch;
  int insize;
  int outwidth, outpitch, outsize;
  int mirw, mirh;
  int nblocks;
  float planeBase, maxval, rounder;
};

inline int roundup16(int v) { return ((v + 15) / 16) * 16; }

inline bool compute_geometry(const FilterParams& p, Geometry& g) {
  int bw = p.bw, bh = p.bh, ow = p.ow, oh = p.oh;
  if (ow * 2 > bw) return false; // "ow must be less than bw / 2"
  if (oh * 2 > bh) return false;
  if (ow < 0) ow = bw / 3;
  if (oh < 0) oh = bh / 3;
  g.ow = ow; g.oh = oh;
  const int SW = bw - ow, SH = bh - oh;
  // engine lines 122-123: crop subtracted BEFORE the chroma subsample shift
  g.nox = (((p.planeW - p.l - p.r) >> p.xShift) - ow + (SW - 1)) / SW;
  g.noy = (((p.planeH - p.t - p.b) >> p.yShift) - oh + (SH - 1)) / SH;
  if (g.nox <= 0 || g.noy <= 0) return false;
  // engine lines 126-130: padding by 1 block per side, mirror size = block interval
  g.nox += 2;
  g.noy += 2;
  // FrameToCover (buffer.cpp 562-569): crop shifted for chroma, frame dims shifted
  g.frameW = (p.planeW >> p.xShift) - (p.l >> p.xShift) - (p.r >> p.xShift);
  g.frameH = (p.planeH >> p.yShift) - (p.t >> p.yShift) - (p.b >> p.yShift);
  if (g.frameW <= 0 || g.frameH <= 0) return false;
  g.coverwidth = g.nox * SW + ow;
  g.coverheight = g.noy * SH + oh;
  g.coverpitch = roundup16(g.coverwidth);
  g.insize = roundup16(bw * bh * g.nox * g.noy);
  g.outwidth = bw / 2 + 1;
  g.outpitch = roundup16(g.outwidth);
  g.outsize = g.outpitch * bh * g.nox * g.noy;
  g.mirw = SW;
  g.mirh = SH;
  g.nblocks = g.nox * g.noy;
  g.planeBase = (p.isChroma && p.bpp < 4) ? (float)(1u << (p.bits - 1)) : 0.0f;
  g.maxval = p.bpp == 4 ? 1.0f : (float)((1u << p.bits) - 1u);
  g.rounder = p.bpp == 4 ? 0.0f : 0.5f;
  return true;
}

// apply the bit-depth scaling to sigma-like params (engine ctor lines 98-110)
inline void scale_sigmas(FilterParams& p) {
  float factor;
  switch (p.bpp) {
    case 1: factor = 1.0f; break;
    case 2: factor = (float)(1 << (p.bits - 8)); break;
    default: factor = 1.0f / 255.0f;
  }
  p.sigma *= factor;
  p.sigma2 *= factor;
  p.sigma3 *= factor;
  p.sigma4 *= factor;
  p.smin *= factor;
  p.smax *= factor;
  // note: ht is NOT scaled (matches original)
}

struct Tables {
  std::vector<float> wanxl, wanxr, wanyl, wanyr;
  std::vector<float> wsynxl, wsynxr, wsynyl, wsynyr;
  std::vector<float> wsharpen, wdehalo; // bh x outpitch floats
  float norm;
  float sigmaSquaredNoiseNormed2D, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, ht2n;
  float lowlimit;
  int ow, oh; // resolved (>=0)
};

inline void make_tables(const FilterParams& p, const Geometry& g, Tables& t) {
  int bw = p.bw, bh = p.bh;
  int ow = p.ow < 0 ? bw / 3 : p.ow;
  int oh = p.oh < 0 ? bh / 3 : p.oh;
  t.ow = ow; t.oh = oh;
  t.wanxl.resize(ow); t.wanxr.resize(ow);
  t.wanyl.resize(oh); t.wanyr.resize(oh);
  t.wsynxl.resize(ow); t.wsynxr.resize(ow);
  t.wsynyl.resize(oh); t.wsynyr.resize(oh);

  const float pi = 3.1415926535897932384626433832795f;
  int i;
  if (p.wintype == 0) {
    for (i = 0; i < ow; i++) {
      t.wanxl[i] = cosf(pi * (i - ow + 0.5f) / (ow * 2));
      t.wanxr[i] = cosf(pi * (i + 0.5f) / (ow * 2));
    }
    for (i = 0; i < oh; i++) {
      t.wanyl[i] = cosf(pi * (i - oh + 0.5f) / (oh * 2));
      t.wanyr[i] = cosf(pi * (i + 0.5f) / (oh * 2));
    }
    t.wsynxl = t.wanxl; t.wsynxr = t.wanxr;
    t.wsynyl = t.wanyl; t.wsynyr = t.wanyr;
  } else if (p.wintype == 1) {
    for (i = 0; i < ow; i++) {
      t.wanxl[i] = sqrtf(cosf(pi * (i - ow + 0.5f) / (ow * 2)));
      t.wanxr[i] = sqrtf(cosf(pi * (i + 0.5f) / (ow * 2)));
    }
    for (i = 0; i < oh; i++) {
      t.wanyl[i] = sqrtf(cosf(pi * (i - oh + 0.5f) / (oh * 2)));
      t.wanyr[i] = sqrtf(cosf(pi * (i + 0.5f) / (oh * 2)));
    }
    for (i = 0; i < ow; i++) {
      t.wsynxl[i] = t.wanxl[i] * t.wanxl[i] * t.wanxl[i];
      t.wsynxr[i] = t.wanxr[i] * t.wanxr[i] * t.wanxr[i];
    }
    for (i = 0; i < oh; i++) {
      t.wsynyl[i] = t.wanyl[i] * t.wanyl[i] * t.wanyl[i];
      t.wsynyr[i] = t.wanyr[i] * t.wanyr[i] * t.wanyr[i];
    }
  } else { // wintype == 2
    std::fill(t.wanxl.begin(), t.wanxl.end(), 1.0f);
    std::fill(t.wanxr.begin(), t.wanxr.end(), 1.0f);
    std::fill(t.wanyl.begin(), t.wanyl.end(), 1.0f);
    std::fill(t.wanyr.begin(), t.wanyr.end(), 1.0f);
    for (i = 0; i < ow; i++) {
      t.wsynxl[i] = cosf(pi * (i - ow + 0.5f) / (ow * 2));
      t.wsynxl[i] = t.wsynxl[i] * t.wsynxl[i];
      t.wsynxr[i] = cosf(pi * (i + 0.5f) / (ow * 2));
      t.wsynxr[i] = t.wsynxr[i] * t.wsynxr[i];
    }
    for (i = 0; i < oh; i++) {
      t.wsynyl[i] = cosf(pi * (i - oh + 0.5f) / (oh * 2));
      t.wsynyl[i] = t.wsynyl[i] * t.wsynyl[i];
      t.wsynyr[i] = cosf(pi * (i + 0.5f) / (oh * 2));
      t.wsynyr[i] = t.wsynyr[i] * t.wsynyr[i];
    }
  }

  // sharpen window (engine lines 292-306)
  t.wsharpen.assign((size_t)bh * g.outpitch, 0.0f);
  for (int j = 0; j < bh; j++) {
    int dj = j;
    if (j >= bh / 2) dj = bh - j;
    float d2v = (float)(dj * dj) * (p.svr * p.svr) / ((bh / 2) * (bh / 2));
    float* row = t.wsharpen.data() + (size_t)j * g.outpitch;
    for (int w = 0; w < g.outwidth; w++) {
      float d2 = d2v + (float)(w * w) / ((bw / 2) * (bw / 2));
      row[w] = 1 - expf(-d2 / (2 * p.scutoff * p.scutoff));
    }
  }

  // dehalo window (engine lines 308-334)
  t.wdehalo.assign((size_t)bh * g.outpitch, 0.0f);
  float wmax = 0;
  for (int j = 0; j < bh; j++) {
    int dj = j;
    if (j >= bh / 2) dj = bh - j;
    float d2v = (float)(dj * dj) * (p.svr * p.svr) / ((bh / 2) * (bh / 2));
    float* row = t.wdehalo.data() + (size_t)j * g.outpitch;
    for (int w = 0; w < g.outwidth; w++) {
      float d2 = d2v + (float)(w * w) / ((bw / 2) * (bw / 2));
      row[w] = expf(-0.7f * d2 * p.hr * p.hr) - expf(-d2 * p.hr * p.hr);
      if (row[w] > wmax) wmax = row[w];
    }
  }
  for (int j = 0; j < bh; j++) {
    float* row = t.wdehalo.data() + (size_t)j * g.outpitch;
    for (int w = 0; w < g.outwidth; w++) row[w] /= wmax;
  }

  t.norm = 1.0f / (bw * bh);
  t.sigmaSquaredNoiseNormed2D = p.sigma * p.sigma / t.norm;
  t.sigmaSquaredSharpenMinNormed = p.smin * p.smin / t.norm;
  t.sigmaSquaredSharpenMaxNormed = p.smax * p.smax / t.norm;
  t.ht2n = p.ht * p.ht / t.norm;
  t.lowlimit = (p.beta - 1.0f) / p.beta;
}

// pattern window array, formulas verbatim from the engine ctor (353-368);
// padding columns [outwidth, outpitch) stay 0 (the CPU leaves them uninitialized
// but never reads them)
inline std::vector<float> make_pwin(const FilterParams& p, const Geometry& g) {
  std::vector<float> pwin((size_t)p.bh * g.outpitch, 0.0f);
  for (int j = 0; j < p.bh; j++) {
    const int dj = j < p.bh - j ? j : p.bh - j; // std::min(j, bh - j)
    const float fh = dj * 2.0f / p.bh;
    const float fh2 = fh * fh;
    float* row = pwin.data() + (size_t)j * g.outpitch;
    for (int i = 0; i < g.outwidth; i++) {
      float fw2 = (i * 2.0f / p.bw) * (i * 2.0f / p.bw);
      row[i] = (fh2 + fw2) / (fh2 + fw2 + p.pcutoff * p.pcutoff);
    }
  }
  return pwin;
}

// helper.cpp SigmasToPattern: pattern built from sigma/sigma2/sigma3/sigma4
// when they differ and pfactor == 0 (engine 373-378 forces pfactor = 1)
inline std::vector<float> sigmas_to_pattern(const FilterParams& p, const Geometry& g, float norm) {
  std::vector<float> pat((size_t)p.bh * g.outpitch, 0.0f);
  const float ft2 = sqrtf(0.5f) / 2; // frequency for sigma2
  const float ft3 = sqrtf(0.5f) / 4; // frequency for sigma3
  for (int h = 0; h < p.bh; h++) {
    float* row = pat.data() + (size_t)h * g.outpitch;
    for (int w = 0; w < g.outwidth; w++) {
      float fy = (p.bh - 2.0f * abs(h - p.bh / 2)) / p.bh; // normalized to 1
      float fx = (w * 1.0f) / g.outwidth;                  // normalized to 1
      float f = sqrtf((fx * fx + fy * fy) * 0.5f);         // normalized to 1
      float sigmacur;
      if (f < ft3)
        sigmacur = p.sigma4 + (p.sigma3 - p.sigma4) * f / ft3;
      else if (f < ft2)
        sigmacur = p.sigma3 + (p.sigma2 - p.sigma3) * (f - ft3) / (ft2 - ft3);
      else
        sigmacur = p.sigma + (p.sigma2 - p.sigma) * (1 - f) / (1 - ft2);
      row[w] = sigmacur * sigmacur / norm;
    }
  }
  return pat;
}

// twiddle tables for the Stockham FFT (same as fft_test)
inline std::vector<float> twiddle_table(int N) {
  std::vector<float> t(2 * (N - 1));
  int off = 0;
  for (int ncur = N; ncur >= 2; ncur >>= 1) {
    for (int p = 0; p < ncur / 2; p++) {
      double a = -2.0 * 3.14159265358979323846 * p / ncur;
      t[2 * (off + p)] = (float)std::cos(a);
      t[2 * (off + p) + 1] = (float)std::sin(a);
    }
    off += ncur / 2;
  }
  return t;
}

} // namespace gpu
