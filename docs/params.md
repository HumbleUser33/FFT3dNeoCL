# FFT3DNEOCL — parameter specification

Source of truth: `neo_FFT3D/src/fft3d.hpp` (`Params()` + `Initialize()`) and
`neo_FFT3D/src/fft3d_engine.h` (validation). Names, order, defaults and
semantics are preserved exactly. The original VapourSynth build exposes
`planes` (the `y/u/v` trio is AviSynth-only: `VSEnabled=false`), so this port
exposes `planes` only, matching the original VS behaviour.

| Parameter | Type | Default | Range / semantics |
|---|---|---|---|
| clip | vnode | — | input clip |
| sigma | float | 2.0 | denoise strength (noise std dev, 8-bit units; scaled by 2^(bits-8) internally, and by 1/255 for float clips) |
| beta | float | 1.0 | Wiener noise margin; must be >= 1.0 (throws otherwise); lowlimit = (beta-1)/beta |
| planes | int[] | [0,1,2] | planes to process; others copied from source |
| bw | int | 32 | block width (this port: power of two, 16..128) |
| bh | int | 32 | block height (this port: power of two, 16..128) |
| bt | int | 3 | -1 sharpen only; 0 Kalman (temporal IIR); 1 2D Wiener; 2..5 3D Wiener over 2..5 frames. Range -1..5 (throws otherwise) |
| ow | int | -1 | horizontal overlap; -1 → bw/3; must satisfy 2*ow <= bw (throws otherwise) |
| oh | int | -1 | vertical overlap; -1 → bh/3; must satisfy 2*oh <= bh |
| kratio | float | 2.0 | Kalman motion threshold to sigma ratio |
| sharpen | float | 0.0 | sharpen strength |
| scutoff | float | 0.3 | sharpen cutoff frequency (relative to max) |
| svr | float | 1.0 | sharpen vertical ratio |
| smin | float | 4.0 | sharpen noise limit (8-bit units, scaled like sigma) |
| smax | float | 20.0 | sharpen overshoot limit (scaled like sigma) |
| measure | bool | true | FFTW_MEASURE vs ESTIMATE. **No effect on GPU** (accepted, ignored; documented) |
| interlaced | bool | false | field-separated block cover (fields stacked, second flipped) |
| wintype | int | 0 | 0 half-cosine anal+synth; 1 flat anal + cube synth; 2 flat anal + raised-cosine synth |
| pframe | int | 0 | frame used to estimate noise pattern when pfactor != 0 |
| px, py | int | 0 | pattern block coords; 0,0 = auto-find quietest block |
| pshow | bool | false | output the pattern block only (debug mode). **Rejected by this port** (debug visualization, not needed in production) |
| pcutoff | float | 0.1 | pattern window cutoff frequency |
| pfactor | float | 0.0 | pattern denoise strength; 0 = use sigmas. If sigma2/3/4 differ from sigma and pfactor==0, pattern is built from sigmas and pfactor becomes 1 |
| sigma2 | float | = sigma | mid-frequency noise level (scaled like sigma) |
| sigma3 | float | = sigma | low-frequency noise level |
| sigma4 | float | = sigma | lowest (DC) noise level |
| degrid | float | 1.0 | window grid compensation strength |
| dehalo | float | 0.0 | dehalo strength |
| hr | float | 2.0 | dehalo radius |
| ht | float | 50.0 | dehalo threshold (scaled like sigma via ht2n) |
| l, t, r, b | int | 0 | crop: border pixels copied from source (clamped >= 0; l+r >= width throws) |
| opt | int | 0 | CPU code path selector. **No effect on GPU** (accepted, ignored) |
| ncpu | int | 2 | FFTW thread count. **No effect on GPU** (accepted, ignored) |
| mt | bool | false | plane parallelism hint. **No effect on GPU** (VS threading covers it) |

New parameters (GPU runtime only, documented additions):
`device_id` (int, default 0), `platform_id` (int, default -1 = auto),
`num_streams` (int, default 1). Output must be bit-identical for any value.

## Formats

8/10/12/14/16-bit integer and 32-bit float, Gray/YUV (4:4:4, 4:2:2, 4:2:0)
and planar RGB, matching the original's dispatch in `buffer.cpp`.

## Edge behaviour (must be reproduced)

- Frames are padded by mirroring one block interval per side (`mirw = bw-ow`,
  `mirh = bh-oh`) in `FrameToCover`.
- For `bt/2 > n` or `(bt-1)/2 > numFrames-1-n`, `btcur` falls back to 1
  (2D-only filtering) at clip start/end.
- `bt=0` (Kalman): frame 0 is returned unchanged; state is sequential.
- Output rounding: `(cast_t)(value*norm + 0.5 + planeBase)` truncated cast
  with +0.5 (integer), clamp to [0, max]; float: no rounding, clamp [0,1].
  Chroma integer planes add planeBase = 1<<(bits-1).
