# Porting map — neo_FFT3D → OpenCL 1.2

One entry per pass of the original pipeline (source: `neo_FFT3D/src/*`).
The GPU port keeps stage boundaries identical so each stage can be validated
against CPU dumps (`fft3d_ref --dump-stages`).

## Engine setup (constructor, `fft3d_engine.h`) — stays on CPU (host)

| Item | Source | Notes |
|---|---|---|
| sigma scaling | ctor lines 96-107 | factor = 1 (8b), 1<<(bits-8) (16b int), 1/255 (float). Applies to sigma, sigma2..4, smin, smax |
| ow/oh defaults | 109-112 | -1 → bw/3, bh/3; validate 2*ow<=bw |
| block grid | 119-125 | nox = ceil((W'-ow)/(bw-ow)) + 2, same for noy; W' = cropped width; +2 padding blocks per axis |
| cover geometry | 126-134 | coverwidth = nox*(bw-ow)+ow; coverpitch = align16(coverwidth); mirw = bw-ow, mirh = bh-oh |
| FFT layout | 136-141 | in: noy*nox blocks of bw*bh float, idist=bw*bh; out: outwidth=bw/2+1 complex rows of bh, outpitch=align16(outwidth), odist=outpitch*bh |
| norm | 333 | norm = 1/(bw*bh); FFTW does NOT normalize — norm applied in OverlapToCover |
| sigma constants | 335-340 | sigmaSquaredNoiseNormed2D = sigma²/norm; sigmaMotionNormed; smin²/norm; smax²/norm; ht2n = ht²/norm |
| analysis/synthesis windows | 212-287 | wintype 0/1/2; cosf in float; tables wanxl/r, wanyl/r (analysis), wsynxl/r, wsynyl/r (synthesis) |
| wsharpen table | 290-303 | per (h,w): 1 - exp(-d2/(2*scutoff²)), d2 = dj²*svr²/(bh/2)² + i²/(bw/2)² |
| wdehalo table | 306-331 | exp(-0.7*d2*hr²) - exp(-d2*hr²), normalized by max |
| pwin table | 350-365 | (fh²+fw²)/(fh²+fw²+pcutoff²) |
| pattern from sigmas | 370-375 + helper.cpp SigmasToPattern | ft2=sqrt(0.5)/2, ft3=sqrt(0.5)/4; piecewise-linear sigma(f); pattern2d = sigmacur²/norm |
| gridsample | 388-408 | cover filled with max pixel value (255 / (2^bits-1) / 1.0f), CoverToOverlap + one r2c FFT of a single block → window compensation spectrum |
| Kalman init | 343-348 | outLast=0, covar=covarProcess=sigma²/norm (complex arrays, both components) |

## Per-frame pipeline (`GetFrame`)

### 1. FrameToCover (`buffer.cpp`) — GPU kernel
- Input: frame plane (u8/u16/f32), cropped region (l,t,r,b applied).
- Output: cover buffer coverwidth×coverpitch, mirror padding: horizontal
  `cover[w]=cover[2*mirw-w]` left, `cover[w]=cover[width2-w]` right; vertical
  mirror line-by-line; interlaced: fields stacked, second field flipped.
- Element type = pixel type (no conversion).

### 2. CoverToOverlap (`buffer.cpp`) — GPU kernel
- Input: cover buffer; Output: float `in[]` of nox*noy blocks (bw×bh, idist=bw*bh).
- Each source sample multiplied by analysis window product (wanxl/r × wanyl/r),
  written to 1, 2 or 4 block positions (overlap). Integer input: subtract
  planeBase (chroma 1<<(bits-1)); float: subtract 0.
- Window product order matters: e.g. `ftmp = float(wanyl[h]*(src-base))` then
  `inp[w] = ftmp*wanxr[w]` — reproduce exactly (see lines 139-378).

### 3. Forward FFT r2c per block (FFTW `plan_many_dft_r2c`) — GPU FFT kernel
- Batched 2D real→complex, bh×bw per block, nox*noy blocks.
- Output packed half-spectrum: bh rows × (bw/2+1) complex, row pitch outpitch.
- GPU design: full complex FFT (Stockham radix-2/4, local memory) then keep
  columns 0..bw/2. Twiddles precomputed on host in double → float.
- Numerics differ from FFTW by design (±1 LSB target).

### 4. Frequency-domain filter — GPU kernels (formulas verbatim)
- `Apply2D` (bt=1): `code_impl/Apply.cpp` lines 13-65. degrid correction
  `gridfraction = degrid*in[2][0][0]/gridsample[0][0]` computed per block
  (loop_wrapper_C line 64); psd = cr²+ci²+1e-15; factor = max((psd-noise)/psd, lowlimit);
  s_fact/d_fact as in source; out = c*factor + gridcorrection.
- `Apply3D2..5` (bt=2..5): hand-written temporal DFT (2/3/4/5 points) across
  the per-frame FFT buffers, Wiener per temporal bin (pattern3d = pattern2d*bt),
  inverse temporal DFT, ×1/bt. Constants: sin120, sin72, cos72, sin144, cos144
  (float literals in source — copy the literals).
- `Kalman` (bt=0): `code_impl/Kalman.cpp`; state outLast/covar/covarProcess
  persist across frames (GPU buffers, sequential frames, parallel bins).
- `Sharpen` (after Wiener/Kalman, and alone for bt=-1): `code_impl/Sharpen.cpp`.
- Pattern estimation (`helper.cpp` FindPatternBlock/SetPattern/PutPatternOnly):
  reduction over blocks + argmin; runs once per clip (guarded by isPatternSet).

### 5. Inverse FFT c2r — GPU FFT kernel
- Hermitian reconstruction + inverse complex FFT; FFTW c2r does NOT divide by
  N² — the 1/(bw*bh) norm is applied in OverlapToCover. Match that.

### 6. OverlapToCover + CoverToFrame (`buffer.cpp`) — GPU kernel(s)
- Synthesis windows applied in overlap regions (see exact products, lines
  381-560), multiply by norm, +0.5, truncating cast, clamp [0,max], +planeBase.
- CoverToFrame: strip mirror padding (interlaced: un-flip second field).

## Host-side additions
- gridsample: computed once per engine with the same cover-fill + FFT path
  (reuse GPU kernels, read back gridsample once).
- pattern mode: pattern2d read back once (or kept on GPU).
- pshow: PutPatternOnly path.

## CPU-only parameters (accepted, ignored)
`opt`, `ncpu`, `measure`, `mt`.
