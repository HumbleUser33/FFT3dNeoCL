# fft3dneocl — Parameter Manual (English)

How to use every parameter of `core.fft3dneocl.FFT3D()`. All names, order and
defaults are identical to the original `neo-fft3d` plugin.

The filter is called from a VapourSynth script (StaxRip's custom filter line or
a `.vpy` file), with keyword arguments:

```python
clip = core.fft3dneocl.FFT3D(clip, sigma=3, bt=4, bw=32, bh=32,
                             ow=16, oh=16, sharpen=1.5)
```

## How the filter works (one paragraph)

Each frame is cut into overlapping rectangles (`bw` × `bh` blocks, sliding by
`bw-ow` pixels). Every block is converted to frequencies (FFT). In the
frequency domain, frequencies that look like noise get reduced (Wiener filter),
while frequencies that look like real detail are kept. The blocks are converted
back to pixels and stitched together with cross-faded edges. For `bt` ≥ 2 the
same is done across several consecutive frames at once, which is much stronger
than per-frame filtering. `sharpen`, `dehalo` and `degrid` work as extra
frequency-domain passes after the denoise.

---

## Core denoise

### `sigma` — denoise strength
- **Type / default:** float, `2.0`
- **Values:** 0 = off; 1–4 typical for video; 5+ very strong. Units are noise
  amplitude on an 8-bit scale (0–255); the plugin rescales it automatically for
  10/12/14/16-bit and float clips.
- **Effect on the image:** how much grain/noise disappears. Too low leaves
  noise; too high erases fine texture (skin, film grain, fabric) and makes
  faces look plastic.
- **Inside the filter:** it is the noise level the Wiener filter subtracts from
  every frequency: a block's frequency is kept in proportion to how much energy
  it has *above* `sigma`². One value for the whole frequency range (see
  `sigma2`–`sigma4` and `pfactor` for a shaped noise model).

### `beta` — denoise aggressiveness floor
- **Type / default:** float, `1.0`
- **Values:** must be ≥ 1.0 (anything else is an error). 1 = maximum strength,
  2–4 gentler.
- **Effect on the image:** at 1, a frequency judged 100 % noise is fully
  deleted. Raising `beta` keeps a guaranteed minimum of every frequency
  (at `beta=2` never less than half), so some residual noise survives but
  textures are less likely to be destroyed.
- **Inside the filter:** sets the lower clamp of the Wiener gain:
  `lowlimit = (beta-1)/beta`. The per-frequency factor is
  `max((energy − noise) / energy, lowlimit)`.

### `bt` — temporal mode (how many frames are used)
- **Type / default:** int, `3`
- **Values:** `-1`, `0`, `1`, `2`, `3`, `4`, `5`.
  - `-1`: no denoise at all — sharpening pass only.
  - `0`: Kalman filter (recursive, one frame at a time).
  - `1`: spatial 2D Wiener only (no temporal).
  - `2`–`5`: 3D Wiener across `bt` consecutive frames (centered window:
    `bt=3` uses previous, current, next).
- **Effect on the image:** higher `bt` averages more frames = much stronger
  denoise in static areas, but moving objects can trail/ghost; low `bt` (or 1)
  is safer on fast motion. `0` (Kalman) accumulates a running average: very
  clean on static content, and frame 0 is returned unchanged (it needs one
  frame to start the memory). Note: at the start and end of the clip, where the
  window doesn't fit, the filter automatically falls back to 2D-only.
- **Inside the filter:** for 2–5, the FFTs of each frame in the window get a
  small temporal transform; each temporal frequency band gets its own Wiener
  factor, then the transform is inverted. For 0, GPU buffers hold the running
  state (value + uncertainty) of every frequency between frames.

### `planes` — which planes to filter
- **Type / default:** int array, `[0,1,2]` (all planes)
- **Values:** any subset of plane indices: `0` = luma (or R in RGB), `1`/`2` =
  chroma U/V (or G/B). Example: `planes=[0]` filters luma only.
- **Effect on the image:** unprocessed planes are copied unchanged from the
  source. Luma-only is faster and keeps chroma detail; processing chroma too
  cleans colored noise but can smear color detail.
- **Inside the filter:** one independent GPU pipeline per processed plane.

---

## Block geometry & reconstruction

### `bw`, `bh` — block size
- **Type / default:** int, `32` × `32`
- **Values:** this GPU port accepts powers of two only: 16, 32, 64, 128.
- **Effect on the image:** bigger blocks see a wider area = better at telling
  noise from detail, but they blur moving objects over a larger patch and use
  more memory/time. 32 is the sweet spot for 1080p/4K; 16 reacts better to
  motion; 64–128 only for slow, very noisy content.
- **Inside the filter:** it is the FFT size per block. Frequency resolution is
  1/`bw` of the block, and all noise thresholds (`sigma`, `smin`, `smax`,
  `ht`, …) are normalized by the block area.

### `ow`, `oh` — block overlap
- **Type / default:** int, `-1` (= `bw/3`, i.e. about one third of the block —
  10 when bw=32)
- **Values:** 0 up to half the block (`2*ow <= bw`), typically 8–16 for 32.
- **Effect on the image:** how much neighbouring blocks cross-fade. More
  overlap = smoother result, fewer visible block edges, but slower (more
  blocks). Too little overlap can leave a faint mosaic on gradients.
- **Inside the filter:** blocks step by `bw-ow` pixels; the overlap areas are
  blended with synthesis windows (see `wintype`).

### `wintype` — window shape
- **Type / default:** int, `0`
- **Values:** `0`: half-cosine analysis + synthesis (default, safest);
  `1`: flat analysis + cubic synthesis; `2`: flat analysis + raised-cosine
  synthesis.
- **Effect on the image:** subtle — how energy near block borders is weighted
  when stitching. 0 is the standard choice; 1 and 2 (flat analysis: the block
  center is untouched) can keep slightly sharper fine detail, at a marginally
  higher chance of edge artifacts.
- **Inside the filter:** the analysis window is multiplied onto each block
  before the FFT, the synthesis windows cross-fade the overlap regions after
  the inverse FFT.

### `interlaced` — field-aware tiling
- **Type / default:** int, `0` (progressive)
- **Values:** 0 or 1.
- **Effect on the image:** for true interlaced sources, blocks are kept inside
  one field, so the filter never averages a top line with the bottom line of
  the 2-line gap between fields (no comb ghosting).
- **Inside the filter:** the frame is tiled with fields stacked (second field
  flipped), and un-flipped before writing back.

### `degrid` — grid compensation
- **Type / default:** float, `1.0`
- **Values:** 0 = off, 1 = full (recommended); >1 over-corrects (not useful).
- **Effect on the image:** cancels a faint regular block-pattern ("grid") that
  heavy denoising can leave on flat areas, because windowed blocks never sum to
  exactly 1 everywhere.
- **Inside the filter:** a flat test pattern goes through the same block/FFT
  path once; the resulting grid spectrum is used to correct every block's DC
  level proportionally to `degrid`.

---

## Kalman (`bt=0` only)

### `kratio` — Kalman motion trust
- **Type / default:** float, `2.0`
- **Values:** 1–5 typical; must be positive.
- **Effect on the image:** how fast the temporal memory gives up on a block
  that changes. Lower = longer memory, cleaner static areas, more ghosting on
  motion. Higher = adapts quickly, less ghosting, weaker temporal denoise.
- **Inside the filter:** the difference between the current block and the
  accumulated state is compared against `sigma × kratio`; beyond it the filter
  assumes change/motion and trusts the new frame more.

---

## Sharpening pass (after the denoise; alone when `bt=-1`)

### `sharpen` — sharpening strength
- **Type / default:** float, `0.0` (off)
- **Values:** 0–2 practical; 1.0–1.5 typical after a strong denoise (which
  softens the image).
- **Effect on the image:** restores perceived crispness lost to denoising.
  Too much creates white halos around edges and re-amplifies any leftover
  noise.
- **Inside the filter:** each frequency of the block is multiplied by a gain
  `1 + sharpen × wsharpen(f) × curve(energy)`, where `wsharpen(f)` rises with
  frequency starting around `scutoff`, and `curve(energy)` (driven by `smin`
  and `smax`) makes the boost fade out on very weak frequencies (noise) and on
  very strong ones (already-sharp edges) — only "detail-level" frequencies get
  the full boost.

### `scutoff` — sharpening frequency cutoff
- **Type / default:** float, `0.3`
- **Values:** 0.1–0.6; a fraction of the maximum frequency.
- **Effect on the image:** low = sharpening reaches down into fine detail
  (stronger, riskier); high = only bigger edges are boosted (softer, safer).
- **Inside the filter:** `wsharpen = 1 − exp(−d² / (2·scutoff²))` — a rising
  frequency curve whose steepness is set by `scutoff`.

### `svr` — sharpening vertical/horizontal balance
- **Type / default:** float, `1.0` (equal)
- **Values:** 0.5–2 practical.
- **Effect on the image:** >1 boosts detail that varies vertically (horizontal
  edges) more than vertical detail; <1 the other way. Useful on material with
  one dominant direction (interlaced upscales, screen content).
- **Inside the filter:** multiplies the vertical frequency coordinate inside
  the `wsharpen` distance.

### `smin`, `smax` — sharpening amplitude guards
- **Type / default:** `4.0` and `20.0` (8-bit units, scaled like `sigma`)
- **Values:** `smin` 2–8, `smax` 10–30 typical; `smin < smax`.
- **Effect on the image:** `smin` = detail quieter than this is not sharpened
  (protects residual noise from being boosted); `smax` = hard limit on the
  boost, preventing ringing/halos on strong edges.
- **Inside the filter:** they set the energy curve of the sharpen gain:
  frequencies quieter than `smin` get almost no boost, frequencies stronger
  than `smax` get almost no boost either (halo prevention); the boost peaks
  between the two.

---

## Dehalo pass

### `dehalo` — dehalo strength
- **Type / default:** float, `0.0` (off)
- **Values:** 0–1 typical, up to ~2.
- **Effect on the image:** thins the bright/dark fringes that appear around
  high-contrast edges (common after sharpening or from source compression).
  Too much makes edges look bruised/dark-outlined.
- **Inside the filter:** an edge-centered high-pass profile (below) is
  subtracted from the block in the frequency domain, scaled by `dehalo`.

### `hr` — dehalo radius
- **Type / default:** float, `2.0`
- **Values:** 1–4; in pixels.
- **Effect on the image:** how far from the edge the halo is sought. Thin
  halos: ~1–2; thick halos from heavy upscaling: 3–4.
- **Inside the filter:** the width of the subtracted profile `wdehalo`,
  normalized to peak at the edge.

### `ht` — dehalo threshold
- **Type / default:** float, `50.0` (8-bit units, scaled like `sigma`)
- **Values:** 20–100.
- **Effect on the image:** only edges stronger than this get dehaloed, so flat
  areas and soft detail are untouched. Lower = applies to more edges (also to
  noise).
- **Inside the filter:** the edge-energy mask uses `ht²`, normalized by the
  block area like the other thresholds.

---

## Noise-pattern mode (advanced)

A flat `sigma` treats all frequencies as equally noisy, which is false for real
film grain or camera noise (usually heavier in the high frequencies). These
options give the filter a *measured or shaped* noise spectrum so it can stop
earlier on the frequencies where there is no noise — more detail preserved at
the same apparent clean-up.

### `pfactor` — pattern denoise strength
- **Type / default:** float, `0.0` (off)
- **Values:** 0 = off (or automatic, see below); 0.1–1.0 to blend in the
  measured spectrum; 1.0 = trust the measurement fully.
- **Effect on the image:** denoises grainy footage while keeping more real
  texture, because the noise model matches the actual noise shape.
- **Inside the filter:** when non-zero, the noise spectrum is estimated once
  per clip from the quietest-looking block of frame `pframe`, windowed by
  `pcutoff`, and used instead of the flat `sigma` model. Special case: if you
  set `sigma2/3/4` differently from `sigma` and leave `pfactor=0`, the pattern
  is built from those sigmas and `pfactor` is forced to 1.

### `pframe` — pattern source frame
- **Type / default:** int, `0`
- **Values:** any frame number (clamped to the clip).
- **Effect:** which frame the noise is measured on. Pick a frame with a flat
  dark area (night sky, shadow wall). If that frame is mid-credits or a
  white flash, the pattern is garbage — choose another.
- **Inside the filter:** the frame is fetched once, and the block whose
  spectrum is quietest is analysed on the first filtered frame.

### `px`, `py` — pattern block position
- **Type / default:** `0, 0` = auto
- **Values:** 0,0 (auto-find the quietest block) or explicit block grid
  coordinates to force one specific block.
- **Effect:** only matters when the automatic block finds something wrong (a
  logo, a dark gradient); then point it at a known-clean area.

### `pcutoff` — pattern frequency window
- **Type / default:** float, `0.1`
- **Values:** 0.01–0.5; a fraction of the frequency range.
- **Effect on the image:** how fast the measured pattern is trusted as
  frequency rises. Small = the pattern mostly shapes the low/mid bands; larger
  = it also dominates the high band.
- **Inside the filter:** window `f² / (f² + pcutoff²)` applied to the
  estimated spectrum.

### `sigma2`, `sigma3`, `sigma4` — per-band noise levels
- **Type / default:** each = `sigma`
- **Values:** same scale as `sigma` (8-bit units, auto-scaled).
- **Effect on the image:** a cheap way to shape the noise model without
  measuring: `sigma2` = mid frequencies, `sigma3` = low frequencies,
  `sigma4` = the very lowest (DC). Example: film grain — `sigma=3, sigma2=3.5,
  sigma3=2, sigma4=1`: denoise the highs hard, stop early on the lows so large
  tonal shapes survive.
- **Inside the filter:** a piecewise-linear noise curve through
  (sigma4, sigma3, sigma2, sigma) over the frequency range; setting them
  different from `sigma` activates pattern mode automatically (`pfactor→1`).

---

## Cropping

### `l`, `t`, `r`, `b` — untouched borders
- **Type / default:** int, `0`
- **Values:** pixels (negative → clamped to 0; cropping the whole width or
  height is an error).
- **Effect on the image:** the border strips are copied pixel-for-pixel from
  the source. Use them to protect credits, station bugs, or black-bar edges
  from temporal smearing.
- **Inside the filter:** the block tiling starts at the cropped region; the
  filter never writes outside it. Chroma borders are rescaled for 4:2:0/4:2:2.

---

## Accepted but ignored (CPU-only parameters)

Kept so existing scripts don't break. They change nothing on the GPU:

| Parameter | Original purpose |
|---|---|
| `measure` | FFTW planning strategy — GPU kernels need no planning |
| `opt` | CPU code-path selector (SSE/AVX variants) |
| `ncpu` | FFTW thread count |
| `mt` | CPU plane-parallelism hint (VapourSynth threads already do this) |

`pshow` (debug view of the pattern block) is **rejected with a clear error** —
not implemented in the port.

---

## GPU selection (this port only)

### `device_id` — which OpenCL device to use
- **Type / default:** int, `0`
- **Values:** device index as listed by `print(core.fft3dneocl.ListDevices())`.
- **Effect:** pick the graphics card when several exist (e.g. dedicated GPU vs
  iGPU). Never changes the pixels.

### `platform_id` — OpenCL platform
- **Type / default:** int, `-1` (auto-pick the best OpenCL 1.2+ device)
- **Values:** platform index; usually left at -1. Only needed to force a
  specific vendor driver when several are installed.

### `num_streams` — internal GPU queues
- **Type / default:** int, `1`
- **Values:** accepted for future use; the result is identical for any value.

---

## Input formats

8/10/12/14/16-bit integer and 32-bit float; Gray, YUV (4:4:4, 4:2:2, 4:2:0)
and planar RGB. 16-bit float is not accepted.

## Quick recipes

| Goal | Call |
|---|---|
| Light clean-up | `sigma=1.5, bt=2` |
| Standard strong denoise | `sigma=2–3, bt=3–4, sharpen=0.5–1.5` |
| Very noisy, little motion | `sigma=4, bt=5, ow=16, oh=16` |
| Only sharpen (no denoise) | `bt=-1, sharpen=1.0` |
| Max temporal (accept ghost risk) | `bt=0, sigma=2, kratio=2` |
| Film grain respect | `sigma=3, sigma2=3.5, sigma3=2, sigma4=1, bt=3` |
| Luma only, fast | `planes=[0], bt=3, sigma=2` |

All defaults already match neo-fft3d, so the shortest valid call is simply:

```python
clip = core.fft3dneocl.FFT3D(clip)   # sigma=2.0, bt=3, bw=bh=32, ...
```
