// FFT3DNEOCL — OpenCL 1.2 kernels (port of neo_FFT3D).
// Reproducibility: FP_CONTRACT off, no fast math, fixed-order reductions,
// twiddle tables precomputed on host in double and rounded to float.

#pragma OPENCL FP_CONTRACT OFF

#ifndef BW
#define BW 32
#endif
#ifndef BH
#define BH 32
#endif

// local pitch padded +1 to break power-of-two bank conflicts on column passes
#define LPITCH (BW + 1)

// fused single-kernel FFT (one work-group per block) compiled only if host says so
#ifndef FFT_FUSED
#define FFT_FUSED 1
#endif

inline float2 cmul(float2 a, float2 b) {
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------
// Forward 2D FFT per block: real input (windowed overlap blocks), half-spectrum
// output in FFTW r2c layout (bh rows x (bw/2+1) complex, row pitch outpitch).
// Stockham autosort radix-2, ping-pong in local memory, one work-group/block.
// twR: row twiddles, packed per stage: stage k (ncur = BW>>k) at offset BW-ncur,
//      entries exp(-2*pi*i*p/ncur), p in [0, ncur/2). Total BW-1 entries.
// twC: same for BH.
// Compiled only when the block fits in device local memory (FFT_FUSED=1).
// ---------------------------------------------------------------------------
#if FFT_FUSED
__kernel void fft_fwd(__global const float* restrict in,
                      __global float2* restrict out,
                      __global const float2* restrict twR,
                      __global const float2* restrict twC,
                      int outpitch)
{
  __local float2 A[BH * LPITCH];
  __local float2 B[BH * LPITCH];

  int block = get_group_id(0);
  int lid = get_local_id(0);
  int wg = get_local_size(0);

  __global const float* bin = in + (size_t)block * BH * BW;

  for (int i = lid; i < BH * BW; i += wg) {
    int r = i / BW, c = i % BW;
    A[r * LPITCH + c] = (float2)(bin[i], 0.0f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- row pass: BH FFTs of length BW ----
  float2 __local * X = A;
  float2 __local * Y = B;
  for (int ncur = BW, s = 1; ncur >= 2; ncur >>= 1, s <<= 1) {
    int m = ncur >> 1;
    int toff = BW - ncur;
    for (int idx = lid; idx < BH * (BW >> 1); idx += wg) {
      int r = idx / (BW >> 1);
      int rem = idx % (BW >> 1);
      int q = rem % s;
      int p = rem / s;
      int base = r * LPITCH;
      float2 a = X[base + q + s * p];
      float2 b = X[base + q + s * (p + m)];
      float2 w = twR[toff + p];
      float2 t = (float2)(a.x - b.x, a.y - b.y);
      Y[base + q + 2 * s * p] = (float2)(a.x + b.x, a.y + b.y);
      Y[base + q + s * (2 * p + 1)] = cmul(t, w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 __local * tmp = X; X = Y; Y = tmp;
  }

  // ---- column pass: BW FFTs of length BH ----
  for (int ncur = BH, s = 1; ncur >= 2; ncur >>= 1, s <<= 1) {
    int m = ncur >> 1;
    int toff = BH - ncur;
    for (int idx = lid; idx < BW * (BH >> 1); idx += wg) {
      int c = idx / (BH >> 1);
      int rem = idx % (BH >> 1);
      int q = rem % s;
      int p = rem / s;
      float2 a = X[c + (q + s * p) * LPITCH];
      float2 b = X[c + (q + s * (p + m)) * LPITCH];
      float2 w = twC[toff + p];
      float2 t = (float2)(a.x - b.x, a.y - b.y);
      Y[c + (q + 2 * s * p) * LPITCH] = (float2)(a.x + b.x, a.y + b.y);
      Y[c + (q + s * (2 * p + 1)) * LPITCH] = cmul(t, w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 __local * tmp = X; X = Y; Y = tmp;
  }

  // ---- write half-spectrum ----
  for (int i = lid; i < BH * (BW / 2 + 1); i += wg) {
    int h = i / (BW / 2 + 1);
    int w = i % (BW / 2 + 1);
    out[(size_t)block * outpitch * BH + h * outpitch + w] = X[h * LPITCH + w];
  }
}

// ---------------------------------------------------------------------------
// Inverse 2D FFT per block: half-spectrum input (FFTW c2r layout), real output
// (no 1/N^2 scaling — the original applies norm in OverlapToCover).
// Method: conj input, forward FFT, take real part (equals unnormalized ifft).
// DC and Nyquist imaginary parts are ignored, matching FFTW c2r semantics.
// ---------------------------------------------------------------------------
__kernel void fft_inv(__global const float2* restrict in,
                      __global float* restrict out,
                      __global const float2* restrict twR,
                      __global const float2* restrict twC,
                      int outpitch)
{
  __local float2 A[BH * LPITCH];
  __local float2 B[BH * LPITCH];

  int block = get_group_id(0);
  int lid = get_local_id(0);
  int wg = get_local_size(0);

  __global const float2* bin = in + (size_t)block * outpitch * BH;

  // load full Hermitian spectrum, conjugated
  for (int i = lid; i < BH * BW; i += wg) {
    int h = i / BW, w = i % BW;
    float2 z;
    if (w <= BW / 2) {
      z = bin[h * outpitch + w];
    } else {
      // 2D Hermitian symmetry: Z[h][w] = conj(Z[(BH-h) mod BH][BW-w])
      int hh = (BH - h) % BH;
      z = bin[hh * outpitch + (BW - w)];
      z.y = -z.y;
    }
    // FFTW c2r ignores imaginary only at the four self-conjugate corners
    if ((w == 0 || w == BW / 2) && (h == 0 || h == BH / 2)) z.y = 0.0f;
    // conjugate for the ifft-via-fft trick
    z.y = -z.y;
    A[h * LPITCH + w] = z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float2 __local * X = A;
  float2 __local * Y = B;
  for (int ncur = BW, s = 1; ncur >= 2; ncur >>= 1, s <<= 1) {
    int m = ncur >> 1;
    int toff = BW - ncur;
    for (int idx = lid; idx < BH * (BW >> 1); idx += wg) {
      int r = idx / (BW >> 1);
      int rem = idx % (BW >> 1);
      int q = rem % s;
      int p = rem / s;
      int base = r * LPITCH;
      float2 a = X[base + q + s * p];
      float2 b = X[base + q + s * (p + m)];
      float2 w = twR[toff + p];
      float2 t = (float2)(a.x - b.x, a.y - b.y);
      Y[base + q + 2 * s * p] = (float2)(a.x + b.x, a.y + b.y);
      Y[base + q + s * (2 * p + 1)] = cmul(t, w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 __local * tmp = X; X = Y; Y = tmp;
  }

  for (int ncur = BH, s = 1; ncur >= 2; ncur >>= 1, s <<= 1) {
    int m = ncur >> 1;
    int toff = BH - ncur;
    for (int idx = lid; idx < BW * (BH >> 1); idx += wg) {
      int c = idx / (BH >> 1);
      int rem = idx % (BH >> 1);
      int q = rem % s;
      int p = rem / s;
      float2 a = X[c + (q + s * p) * LPITCH];
      float2 b = X[c + (q + s * (p + m)) * LPITCH];
      float2 w = twC[toff + p];
      float2 t = (float2)(a.x - b.x, a.y - b.y);
      Y[c + (q + 2 * s * p) * LPITCH] = (float2)(a.x + b.x, a.y + b.y);
      Y[c + (q + s * (2 * p + 1)) * LPITCH] = cmul(t, w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 __local * tmp = X; X = Y; Y = tmp;
  }

  // real part of conj(result) == real part of result
  __global float* bout = out + (size_t)block * BH * BW;
  for (int i = lid; i < BH * BW; i += wg) {
    int h = i / BW, w = i % BW;
    bout[i] = X[h * LPITCH + w].x;
  }
}
#endif // FFT_FUSED

// ---------------------------------------------------------------------------
// Split FFT for blocks that do not fit in local memory: one work-group per
// row / column, full-complex intermediate in global tmp (row pitch BW).
// Same Stockham stages and twiddle tables as the fused kernels.
// ---------------------------------------------------------------------------

// 1D Stockham in local memory; returns the array holding the result.
inline float2 __local * fft1d(__local float2 *X, __local float2 *Y, int N,
                              __global const float2 *tw, int lid, int wg)
{
  for (int ncur = N, s = 1; ncur >= 2; ncur >>= 1, s <<= 1) {
    int m = ncur >> 1;
    int toff = N - ncur;
    for (int idx = lid; idx < N >> 1; idx += wg) {
      int q = idx % s;
      int p = idx / s;
      float2 a = X[q + s * p];
      float2 b = X[q + s * (p + m)];
      float2 w = tw[toff + p];
      float2 t = (float2)(a.x - b.x, a.y - b.y);
      Y[q + 2 * s * p] = (float2)(a.x + b.x, a.y + b.y);
      Y[q + s * (2 * p + 1)] = cmul(t, w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 __local * tmp = X; X = Y; Y = tmp;
  }
  return X;
}

// forward, row pass: real in -> full complex tmp. Grid: nblocks*BH groups.
__kernel void fft_fwd_rows(__global const float* restrict in,
                           __global float2* restrict tmp,
                           __global const float2* restrict twR)
{
  __local float2 A[BW];
  __local float2 B[BW];
  int row = get_group_id(0);
  int lid = get_local_id(0);
  int wg = get_local_size(0);
  __global const float* src = in + (size_t)row * BW;
  for (int i = lid; i < BW; i += wg) A[i] = (float2)(src[i], 0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  float2 __local * R = fft1d(A, B, BW, twR, lid, wg);
  __global float2* dst = tmp + (size_t)row * BW;
  for (int i = lid; i < BW; i += wg) dst[i] = R[i];
}

// forward, column pass: tmp -> half-spectrum out. Grid: nblocks*(BW/2+1).
__kernel void fft_fwd_cols(__global const float2* restrict tmp,
                           __global float2* restrict out,
                           __global const float2* restrict twC,
                           int outpitch)
{
  __local float2 A[BH];
  __local float2 B[BH];
  int col = get_group_id(0);
  int lid = get_local_id(0);
  int wg = get_local_size(0);
  int block = col / (BW / 2 + 1);
  int w = col % (BW / 2 + 1);
  __global const float2* src = tmp + (size_t)block * BH * BW + w;
  for (int i = lid; i < BH; i += wg) A[i] = src[(size_t)i * BW];
  barrier(CLK_LOCAL_MEM_FENCE);
  float2 __local * R = fft1d(A, B, BH, twC, lid, wg);
  __global float2* dst = out + (size_t)block * outpitch * BH + w;
  for (int i = lid; i < BH; i += wg) dst[(size_t)i * outpitch] = R[i];
}

// inverse, row pass: half-spectrum -> conj-loaded full tmp. Grid: nblocks*BH.
__kernel void fft_inv_rows(__global const float2* restrict bin,
                           __global float2* restrict tmp,
                           __global const float2* restrict twR,
                           int outpitch)
{
  __local float2 A[BW];
  __local float2 B[BW];
  int row = get_group_id(0);
  int lid = get_local_id(0);
  int wg = get_local_size(0);
  int block = row / BH;
  int h = row % BH;
  const size_t odist = (size_t)outpitch * BH;
  __global const float2* src = bin + (size_t)block * odist + (size_t)h * outpitch;
  for (int i = lid; i < BW; i += wg) {
    int w = i;
    float2 z;
    if (w <= BW / 2) {
      z = src[w];
    } else {
      // 2D Hermitian symmetry: Z[h][w] = conj(Z[(BH-h) mod BH][BW-w])
      int hh = (BH - h) % BH;
      z = bin[(size_t)block * odist + (size_t)hh * outpitch + (BW - w)];
      z.y = -z.y;
    }
    // FFTW c2r ignores imaginary only at the four self-conjugate corners
    if ((w == 0 || w == BW / 2) && (h == 0 || h == BH / 2)) z.y = 0.0f;
    // conjugate for the ifft-via-fft trick
    z.y = -z.y;
    A[w] = z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float2 __local * R = fft1d(A, B, BW, twR, lid, wg);
  __global float2* dst = tmp + (size_t)row * BW;
  for (int i = lid; i < BW; i += wg) dst[i] = R[i];
}

// inverse, column pass: tmp -> real out (no scaling). Grid: nblocks*BW.
__kernel void fft_inv_cols(__global const float2* restrict tmp,
                           __global float* restrict out,
                           __global const float2* restrict twC)
{
  __local float2 A[BH];
  __local float2 B[BH];
  int col = get_group_id(0);
  int lid = get_local_id(0);
  int wg = get_local_size(0);
  int block = col / BW;
  int w = col % BW;
  __global const float2* src = tmp + (size_t)block * BH * BW + w;
  for (int i = lid; i < BH; i += wg) A[i] = src[(size_t)i * BW];
  barrier(CLK_LOCAL_MEM_FENCE);
  float2 __local * R = fft1d(A, B, BH, twC, lid, wg);
  __global float* dst = out + (size_t)block * BH * BW + w;
  for (int i = lid; i < BH; i += wg) dst[(size_t)i * BW] = R[i].x;
}

// ---------------------------------------------------------------------------
// gridfraction: per-block degrid scale = degrid * out[block][0][0] / gridsample[0][0]
// (computed before apply modifies out). Grid: howmanyblocks.
// ---------------------------------------------------------------------------
__kernel void gridfraction_k(__global const float2* out,
                             __global const float2* gridsample,
                             float degrid, int odist,
                             __global float* gf)
{
  int b = get_global_id(0);
  gf[b] = degrid * out[(size_t)b * odist].x / gridsample[0].x;
}

// ---------------------------------------------------------------------------
// apply2d: Wiener (+pattern) with degrid correction, sharpen and dehalo
// factors, per code_impl/Apply.cpp Apply2D_C_impl. In-place on out.
// Grid: howmanyblocks * bh * outwidth.
// ---------------------------------------------------------------------------
__kernel void apply2d(__global float2* out,
                      __global const float* gf,
                      __global const float* pattern2d,
                      __global const float2* gridsample,
                      __global const float* wsharpen,
                      __global const float* wdehalo,
                      int outpitch, int outwidth, int bh,
                      float sigmaSquaredNoiseNormed, float lowlimit,
                      float sharpen, float dehalo,
                      float minN, float maxN, float ht2n,
                      int use_pattern)
{
  int idx = get_global_id(0);
  int w = idx % outwidth;
  int h = (idx / outwidth) % bh;
  int b = idx / (outwidth * bh);

  const size_t odist = (size_t)outpitch * bh;
  size_t o = (size_t)b * odist + (size_t)h * outpitch + w;
  size_t t = (size_t)h * outpitch + w; // per-block tables

  float gc0 = gf[b] * gridsample[t].x;
  float gc1 = gf[b] * gridsample[t].y;

  float cr = out[o].x - gc0;
  float ci = out[o].y - gc1;

  float psd = cr * cr + ci * ci + 1e-15f;
  float noise = use_pattern ? pattern2d[t] : sigmaSquaredNoiseNormed;
  float factor = fmax((psd - noise) / psd, lowlimit);

  // Apply2D_C_impl / Apply2D_AVX skip sharpen and dehalo for the pattern
  // family (if constexpr (!pattern)); the engine runs Sharpen as a separate
  // pass instead (fft3d_engine.h 697-698).
  if (!use_pattern) {
    // with sharpen==0 / dehalo==0 these are exactly 1.0, matching the original
    float s_fact = 1.0f + sharpen * wsharpen[t] *
      sqrt(psd * maxN / ((psd + minN) * (psd + maxN)));
    float d_fact = (psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[t] * psd);
    factor *= s_fact;
    factor *= d_fact;
  }

  out[o].x = cr * factor + gc0;
  out[o].y = ci * factor + gc1;
}

// ---------------------------------------------------------------------------
// apply3d: temporal DFT over BT frames, Wiener, inverse temporal DFT.
// Transcribed from code_impl/Apply.cpp Apply3D2_C .. Apply3D5_C (pattern=false).
// s0..s4 are the raw spectra of the BT frames; s2 is always the current frame
// (same indexing as the CPU's apply_in[2 + i]). out is a separate buffer.
// BT is a compile-time define (2..5). Grid: nblocks * bh * outwidth.
// ---------------------------------------------------------------------------
#ifndef BT
#define BT 1
#endif

#if BT >= 2
__kernel void apply3d(__global const float2* s0, __global const float2* s1,
                      __global const float2* s2, __global const float2* s3,
                      __global const float2* s4,
                      __global float2* out,
                      __global const float* gf,
                      __global const float2* gridsample,
                      int outpitch, int outwidth, int bh,
                      float sigmaSquaredNoiseNormed3D, float lowlimit,
                      __global const float* pattern2d, int use_pattern)
{
  int idx = get_global_id(0);
  int w = idx % outwidth;
  int h = (idx / outwidth) % bh;
  int b = idx / (outwidth * bh);

  const size_t odist = (size_t)outpitch * bh;
  const size_t t = (size_t)h * outpitch + w;
  const size_t o = (size_t)b * odist + t;

  // pattern mode: the CPU precomputes pattern3d[i] = pattern2d[i] * btcur once
  // (helper.cpp Pattern2Dto3D); the same float multiply done inline here.
  const float noise3d = use_pattern ? pattern2d[t] * (float)BT
                                    : sigmaSquaredNoiseNormed3D;

  // CPU: gridfraction * gridsample[w] * btcur  (left-to-right multiplication)
  float gc0 = (gf[b] * gridsample[t].x) * (float)BT;
  float gc1 = (gf[b] * gridsample[t].y) * (float)BT;

#define WIENER3D(dr, di) \
  do { \
    float psd = (dr) * (dr) + (di) * (di) + 1.0e-15f; \
    float wf = fmax((psd - noise3d) / psd, lowlimit); \
    (dr) *= wf; (di) *= wf; \
  } while (0)

#if BT == 2
  const float2 cur = s2[o];
  const float2 prev = s1[o];
  // dft 3d (2 points)
  float f3d0r = cur.x + prev.x - gc0;
  float f3d0i = cur.y + prev.y - gc1;
  float f3d1r = cur.x - prev.x;
  float f3d1i = cur.y - prev.y;
  WIENER3D(f3d0r, f3d0i);
  WIENER3D(f3d1r, f3d1i);
  // reverse dft for 2 points
  out[o].x = (f3d0r + f3d1r + gc0) * 0.5f;
  out[o].y = (f3d0i + f3d1i + gc1) * 0.5f;

#elif BT == 3
  const float sin120 = 0.86602540378443864676372317075294f; // sqrt(3)*0.5
  const float athird = 1.0f / 3.0f;
  const float2 cur = s2[o];
  const float2 prev = s1[o];
  const float2 next = s3[o];
  // dft 3d (3 points)
  float pnr = prev.x + next.x;
  float pni = prev.y + next.y;
  float fcr = cur.x + pnr - gc0;
  float fci = cur.y + pni - gc1;
  float di = sin120 * (prev.y - next.y);
  float dr = sin120 * (next.x - prev.x);
  float fpr = cur.x - 0.5f * pnr + di;
  float fnr = cur.x - 0.5f * pnr - di;
  float fpi = cur.y - 0.5f * pni + dr;
  float fni = cur.y - 0.5f * pni - dr;
  WIENER3D(fcr, fci);
  WIENER3D(fpr, fpi);
  WIENER3D(fnr, fni);
  // reverse dft for 3 points
  out[o].x = (fcr + fpr + fnr + gc0) * athird;
  out[o].y = (fci + fpi + fni + gc1) * athird;

#elif BT == 4
  const float2 cur = s2[o];
  const float2 prev = s1[o];
  const float2 next = s3[o];
  const float2 prev2 = s0[o];
  float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
  // dft 3d (4 points)
  fp2r = (cur.x + prev2.x) - (prev.x + next.x);
  fp2i = (cur.y + prev2.y) - (prev.y + next.y);
  fpr  = (cur.x - prev2.x) + (prev.y - next.y);
  fpi  = (cur.y - prev2.y) - (prev.x - next.x);
  fcr  = (cur.x + prev2.x) + (prev.x + next.x) - gc0;
  fci  = (cur.y + prev2.y) + (prev.y + next.y) - gc1;
  fnr  = (cur.x - prev2.x) - (prev.y - next.y);
  fni  = (cur.y - prev2.y) + (prev.x - next.x);
  WIENER3D(fp2r, fp2i);
  WIENER3D(fpr, fpi);
  WIENER3D(fcr, fci);
  WIENER3D(fnr, fni);
  // reverse dft for 4 points
  out[o].x = ((fp2r + fpr) + (fcr + fnr) + gc0) * 0.25f;
  out[o].y = ((fp2i + fpi) + (fci + fni) + gc1) * 0.25f;

#else /* BT == 5 */
  const float sin72 = 0.95105651629515357211643933337938f;
  const float cos72 = 0.30901699437494742410229341718282f;
  const float sin144 = 0.58778525229247312916870595463907f;
  const float cos144 = -0.80901699437494742410229341718282f;
  const float2 cur = s2[o];
  const float2 prev = s1[o];
  const float2 next = s3[o];
  const float2 prev2 = s0[o];
  const float2 next2 = s4[o];
  // dft 3d (5 points)
  float sum = (prev2.x + next2.x) * cos72 + (prev.x + next.x) * cos144 + cur.x;
  float dif = (-prev2.y + next2.y) * sin72 + (prev.y - next.y) * sin144;
  float fp2r = sum + dif;
  float fn2r = sum - dif;
  sum = (prev2.y + next2.y) * cos72 + (prev.y + next.y) * cos144 + cur.y;
  dif = (prev2.x - next2.x) * sin72 + (-prev.x + next.x) * sin144;
  float fp2i = sum + dif;
  float fn2i = sum - dif;
  sum = (prev2.x + next2.x) * cos144 + (prev.x + next.x) * cos72 + cur.x;
  dif = (prev2.y - next2.y) * sin144 + (prev.y - next.y) * sin72;
  float fpr = sum + dif;
  float fnr = sum - dif;
  sum = (prev2.y + next2.y) * cos144 + (prev.y + next.y) * cos72 + cur.y;
  dif = (-prev2.x + next2.x) * sin144 + (-prev.x + next.x) * sin72;
  float fpi = sum + dif;
  float fni = sum - dif;
  float fcr = (prev2.x + prev.x) + cur.x + (next.x + next2.x) - gc0;
  float fci = (prev2.y + prev.y) + cur.y + (next.y + next2.y) - gc1;
  WIENER3D(fp2r, fp2i);
  WIENER3D(fpr, fpi);
  WIENER3D(fcr, fci);
  WIENER3D(fnr, fni);
  WIENER3D(fn2r, fn2i);
  // reverse dft for 5 points
  out[o].x = ((fp2r + fpr) + (fcr + fnr) + fn2r + gc0) * 0.2f;
  out[o].y = ((fp2i + fpi) + (fci + fni) + fn2i + gc1) * 0.2f;
#endif

#undef WIENER3D
}
#endif // BT >= 2

// ---------------------------------------------------------------------------
// kalman: the bt=0 filter. Literal transcription of the per-bin math in
// Kalman_AVX/SSE2 (neo_FFT3D/src/code_impl/Kalman_AVX.cpp 41-69) — the SIMD
// loop wrappers the CPU dispatches process every column of the padded block
// (pos < bh*outpitch), mirror columns included, which keeps the spectrum
// Hermitian for the inverse FFT. outLast holds the previous output spectrum
// and is rewritten in place; covar/covarProcess are the persistent state.
// ---------------------------------------------------------------------------
__kernel void kalman(__global float2* outLast,
                     __global float2* covar,
                     __global float2* covarProcess,
                     __global const float2* incur,
                     int outpitch, int bh,
                     float sigma, float kratio2,
                     __global const float* pattern2d, int use_pattern)
{
  int idx = get_global_id(0);
  int w = idx % outpitch;
  int h = (idx / outpitch) % bh;
  int b = idx / (outpitch * bh);

  const size_t odist = (size_t)outpitch * bh;
  size_t o = (size_t)b * odist + (size_t)h * outpitch + w;

  // Kalman_AVX<true> (pattern): sigma = max(pattern2d[bin], 1e-15) per bin
  float sg = use_pattern ? fmax(pattern2d[(size_t)h * outpitch + w], 1.0e-15f) : sigma;

  float2 cur = incur[o];
  float2 prev = outLast[o]; // == outLast[o], read before the write below

  float thr = sg * kratio2;
  float dr = cur.x - prev.x;
  float di = cur.y - prev.y;
  if (dr * dr > thr || di * di > thr) {
    // big pixel variation due to motion etc: reset filter
    covar[o] = (float2)(sg, sg);
    covarProcess[o] = (float2)(sg, sg);
    outLast[o] = cur;
  } else {
    // small variation
    float2 c = covar[o];
    float2 cp = covarProcess[o];
    float sumr = c.x + cp.x;
    float sumi = c.y + cp.y;
    float gainr = sumr / (sumr + sg);
    float gaini = sumi / (sumi + sg);
    covarProcess[o] = (float2)(gainr * gainr * sg, gaini * gaini * sg);
    covar[o] = (float2)((1.0f - gainr) * sumr, (1.0f - gaini) * sumi);
    outLast[o] = (float2)(gainr * cur.x + (1.0f - gainr) * prev.x,
                          gaini * cur.y + (1.0f - gaini) * prev.y);
  }
}

// ---------------------------------------------------------------------------
// sharpen_post: the extra Sharpen pass the engine runs after Apply3D (only
// when sharpen != 0 or dehalo != 0; the CPU returns early otherwise).
// Same body as Sharpen_C_impl, but without the Wiener factor.
// ---------------------------------------------------------------------------
__kernel void sharpen_post(__global float2* out,
                           __global const float* gf,
                           __global const float2* gridsample,
                           __global const float* wsharpen,
                           __global const float* wdehalo,
                           int outpitch, int outwidth, int bh,
                           float sharpen, float dehalo,
                           float minN, float maxN, float ht2n)
{
  int idx = get_global_id(0);
  int w = idx % outwidth;
  int h = (idx / outwidth) % bh;
  int b = idx / (outwidth * bh);

  const size_t odist = (size_t)outpitch * bh;
  const size_t t = (size_t)h * outpitch + w;
  const size_t o = (size_t)b * odist + t;

  float gc0 = gf[b] * gridsample[t].x;
  float gc1 = gf[b] * gridsample[t].y;

  float cr = out[o].x - gc0;
  float ci = out[o].y - gc1;
  float psd = cr * cr + ci * ci + 1.0e-15f;

  float s_fact = 1.0f + sharpen * wsharpen[t] *
    sqrt(psd * maxN / ((psd + minN) * (psd + maxN)));
  float d_fact = (psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[t] * psd);

  out[o].x = cr * s_fact * d_fact + gc0;
  out[o].y = ci * s_fact * d_fact + gc1;
}

// ---------------------------------------------------------------------------
// Buffer passes. BPP = bytes per pixel (1, 2 or 4). All are pure gathers, one
// work-item per pixel; window products follow buffer.cpp semantics.
// ---------------------------------------------------------------------------

#ifndef BPP
#define BPP 1
#endif
#ifndef OW
#define OW 5
#endif
#ifndef OH
#define OH 5
#endif

#if BPP != 4
inline float read_px(__global const uchar* p) {
#if BPP == 1
  return (float)p[0];
#else
  return (float)((uint)p[0] | ((uint)p[1] << 8));
#endif
}

inline void write_px(__global uchar* p, float v) {
#if BPP == 1
  p[0] = (uchar)v;
#else
  int iv = (int)v;
  p[0] = (uchar)(iv & 0xff); p[1] = (uchar)(iv >> 8);
#endif
}
#endif

// float pixel read/write for BPP==4 (bit-accurate, no rounding)
inline float read_px_f(__global const uchar* p) {
  union { uint u; float f; } c;
  c.u = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
  return c.f;
}
inline void write_px_f(__global uchar* p, float v) {
  union { uint u; float f; } c;
  c.f = v;
  p[0] = (uchar)(c.u & 0xff); p[1] = (uchar)((c.u >> 8) & 0xff);
  p[2] = (uchar)((c.u >> 16) & 0xff); p[3] = (uchar)((c.u >> 24) & 0xff);
}

inline float px_read(__global const uchar* p) {
#if BPP == 4
  return read_px_f(p);
#else
  return read_px(p);
#endif
}

// ---------------------------------------------------------------------------
// frame_to_cover: mirror-pad the (cropped) frame plane into the cover buffer.
// Grid: (coverwidth, coverheight).
// ---------------------------------------------------------------------------
__kernel void frame_to_cover(__global const uchar* src, int src_pitch, int W, int H,
                             __global uchar* cover, int coverpitch,
                             int coverwidth, int coverheight, int mirw, int mirh,
                             int interlaced)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= coverwidth || y >= coverheight) return;

  // cover row -> main-region cover row (vertical mirror)
  int fy;
  if (y < mirh) fy = 2 * mirh - y;
  else if (y >= mirh + H) fy = 2 * (mirh + H) - 2 - y;
  else fy = y;
  // main-region cover row -> frame row (interlaced: fields stacked, second flipped)
  int fr;
  if (!interlaced) fr = fy - mirh;
  else {
    int i = fy - mirh;
    fr = (i < H / 2) ? 2 * i : (H - 1 - 2 * (i - H / 2));
  }

  // cover column -> frame column (horizontal mirror)
  int fx;
  if (x < mirw) fx = mirw + mirw - x;
  else if (x >= W + mirw) fx = (2 * W + 2 * mirw - 2) - x;
  else fx = x;
  fx -= mirw;

  size_t sidx = (size_t)fr * src_pitch + (size_t)fx * BPP; // src_pitch in bytes
  size_t didx = (size_t)y * coverpitch * BPP + (size_t)x * BPP; // coverpitch in pixels
#if BPP == 4
  write_px_f(cover + didx, read_px_f(src + sidx));
#else
  for (int b = 0; b < BPP; b++) cover[didx + b] = src[sidx + b];
#endif
}

// ---------------------------------------------------------------------------
// cover_to_overlap: windowed gather from cover into the float block array.
// Block (by,bx) at offset (by*nox+bx)*BH*BW. Grid: (coverwidth, coverheight).
// ---------------------------------------------------------------------------
__kernel void cover_to_overlap(__global const uchar* cover, int coverpitch,
                               __global float* out,
                               __global const float* wanxl, __global const float* wanxr,
                               __global const float* wanyl, __global const float* wanyr,
                               int coverwidth, int coverheight, int nox, int noy,
                               float planeBase)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= coverwidth || y >= coverheight) return;

  const int SW = BW - OW, SH = BH - OH;
  int ihx = x / SW, wx = x - ihx * SW;
  int ihy = y / SH, wy = y - ihy * SH;

  float v = px_read(cover + (size_t)y * coverpitch * BPP + (size_t)x * BPP) - planeBase;

  // horizontal contributions: (block column, local column, window)
  int bx[2], lc[2];
  float hx[2];
  int nx;
  if (wx < OW) {
    nx = 0;
    if (ihx >= 1) { bx[nx] = ihx - 1; lc[nx] = wx + SW; hx[nx] = wanxr[wx]; nx++; }
    if (ihx <= nox - 1) { bx[nx] = ihx; lc[nx] = wx; hx[nx] = wanxl[wx]; nx++; }
  } else {
    nx = 1; bx[0] = ihx; lc[0] = wx; hx[0] = 1.0f;
  }

  int by[2], lr[2];
  float hy[2];
  int ny;
  if (wy < OH) {
    ny = 0;
    if (ihy >= 1) { by[ny] = ihy - 1; lr[ny] = wy + SH; hy[ny] = wanyr[wy]; ny++; }
    if (ihy <= noy - 1) { by[ny] = ihy; lr[ny] = wy; hy[ny] = wanyl[wy]; ny++; }
  } else {
    ny = 1; by[0] = ihy; lr[0] = wy; hy[0] = 1.0f;
  }

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      size_t off = ((size_t)(by[j] * nox + bx[i])) * BH * BW + (size_t)lr[j] * BW + lc[i];
      out[off] = v * hx[i] * hy[j];
    }
}

// ---------------------------------------------------------------------------
// overlap_to_cover: synthesis-window overlap-add back into the cover buffer,
// with norm, +0.5 rounding, truncating cast and clamp (integer) / clamp (float).
// Grid: (coverwidth, coverheight).
// ---------------------------------------------------------------------------
__kernel void overlap_to_cover(__global const float* in,
                               __global uchar* cover, int coverpitch,
                               __global const float* wsynxl, __global const float* wsynxr,
                               __global const float* wsynyl, __global const float* wsynyr,
                               int coverwidth, int coverheight, int nox, int noy,
                               float norm, float rounder, float planeBase, float maxval)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= coverwidth || y >= coverheight) return;

  const int SW = BW - OW, SH = BH - OH;
  int ihx = x / SW, wx = x - ihx * SW;
  int ihy = y / SH, wy = y - ihy * SH;

  int bx[2], lc[2];
  float hx[2];
  int nx;
  if (wx < OW) {
    nx = 0;
    if (ihx >= 1) {
      bx[nx] = ihx - 1; lc[nx] = wx + SW;
      hx[nx] = (ihx == nox) ? 1.0f : wsynxr[wx]; // last block right edge: no synthesis window
      nx++;
    }
    if (ihx <= nox - 1) {
      bx[nx] = ihx; lc[nx] = wx;
      hx[nx] = (ihx == 0) ? 1.0f : wsynxl[wx]; // edge block: no synthesis window (original behaviour)
      nx++;
    }
  } else {
    nx = 1; bx[0] = ihx; lc[0] = wx; hx[0] = 1.0f;
  }

  int by[2], lr[2];
  float hy[2];
  int ny;
  if (wy < OH) {
    ny = 0;
    if (ihy >= 1) {
      by[ny] = ihy - 1; lr[ny] = wy + SH;
      hy[ny] = (ihy == noy) ? 1.0f : wsynyr[wy]; // last block bottom edge: no synthesis window
      ny++;
    }
    if (ihy <= noy - 1) {
      by[ny] = ihy; lr[ny] = wy;
      hy[ny] = (ihy == 0) ? 1.0f : wsynyl[wy];
      ny++;
    }
  } else {
    ny = 1; by[0] = ihy; lr[0] = wy; hy[0] = 1.0f;
  }

  // Reproduce the CPU operation order exactly (bit-exact for float32):
  //  ny==1:  (a*hx0 + b*hx1) * norm
  //  ny==2:  (a*hx0 + b*hx1) * (hy0*norm) + (c*hx0 + d*hx1) * (hy1*norm)
  // (the CPU pre-multiplies the y synthesis windows by norm, see buffer.cpp)
  float expr;
  if (ny == 1) {
    float sum = 0.0f;
    for (int i = 0; i < nx; i++) {
      size_t off = ((size_t)(by[0] * nox + bx[i])) * BH * BW + (size_t)lr[0] * BW + lc[i];
      sum += in[off] * hx[i];
    }
    expr = sum * norm;
  } else {
    float yrh = hy[0] * norm;
    float ylh = hy[1] * norm;
    float sx0 = 0.0f, sx1 = 0.0f;
    for (int i = 0; i < nx; i++) {
      size_t o0 = ((size_t)(by[0] * nox + bx[i])) * BH * BW + (size_t)lr[0] * BW + lc[i];
      size_t o1 = ((size_t)(by[1] * nox + bx[i])) * BH * BW + (size_t)lr[1] * BW + lc[i];
      sx0 += in[o0] * hx[i];
      sx1 += in[o1] * hx[i];
    }
    expr = sx0 * yrh + sx1 * ylh;
  }
  size_t didx = (size_t)y * coverpitch * BPP + (size_t)x * BPP;
#if BPP == 4
  float v = clamp(expr + rounder + planeBase, 0.0f, maxval); // planeBase/rounder are 0 for float
  write_px_f(cover + didx, v);
#else
  int iv;
  if (ny == 2 && nx == 1 && ihx == 0) {
    // CPU quirk (buffer.cpp line 454): in the first segment of the vertical
    // overlap rows, planeBase is added AFTER the truncating cast. Only affects
    // mirror-padding columns (never copied to output), but reproduce it anyway.
    iv = (int)(expr + rounder) + (int)planeBase;
  } else {
    iv = (int)(expr + rounder + planeBase);
  }
  if (iv < 0) iv = 0;
  if (iv > (int)maxval) iv = (int)maxval;
  write_px(cover + didx, (float)iv);
#endif
}

// ---------------------------------------------------------------------------
// cover_to_frame: strip the mirror padding (inverse field mapping for interlaced).
// Grid: (W, H) of the cropped frame.
// ---------------------------------------------------------------------------
__kernel void cover_to_frame(__global const uchar* cover, int coverpitch,
                             __global uchar* dst, int dst_pitch, int W, int H,
                             int mirw, int mirh, int interlaced)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= W || y >= H) return;

  int cy;
  if (!interlaced) cy = mirh + y;
  else cy = (y & 1) ? (mirh + H / 2 + (H - 1 - y) / 2) : (mirh + y / 2);
  int cx = mirw + x;

  size_t sidx = (size_t)cy * coverpitch * BPP + (size_t)cx * BPP; // coverpitch in pixels
  size_t didx = (size_t)y * dst_pitch + (size_t)x * BPP; // dst_pitch in bytes
#if BPP == 4
  write_px_f(dst + didx, read_px_f(cover + sidx));
#else
  for (int b = 0; b < BPP; b++) dst[didx + b] = cover[sidx + b];
#endif
}
