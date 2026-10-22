# FFT3dNeoCL

**neo_FFT3D, running on your GPU.** This repository is a fork of
[neo_FFT3D](https://github.com/HomeOfAviSynthPlusEvolution/neo_FFT3D), the
FFT-based video denoiser, with the whole engine ported to **OpenCL** so it runs
on any modern graphics card (NVIDIA, AMD, Intel) instead of the CPU. The port
ships as a **VapourSynth plugin** (`fft3dneocl.dll`, API 4).

The algorithm is unchanged: same parameters, same defaults, same results. The
GPU output is compared against the original CPU engine and is **bit-for-bit
equal except for rounding** — see [Quality](#quality-vs-the-cpu-engine) below.

## What was done

- Full port of the `FFT3DEngine` pipeline to portable OpenCL C 1.2: 2D/3D FFT,
  Wiener filter, degrid, temporal modes, pattern (motif) processing, Kalman.
- Every mode and format supported: `blocktype` -1 to 5 (including Kalman
  `bt=0` and bm1 `bt=-1`), `pfactor`/pattern, `sharpen`/`dehalo`, crop,
  interlaced, window types, plane selection, 8/10/12/14/16-bit integer and
  32-bit float, GRAY / YUV420 / YUV422 / YUV444.
- A validation harness that runs the original CPU engine and the GPU plugin
  side by side: a 51-fixture regression matrix plus comparisons on real 4K
  video. The CPU engine gained optional compile-time diagnostic hooks (no
  numeric effect) so each pipeline stage can be compared separately.
- GPU pipeline pooling (contexts and buffers are reused between frames and
  previews) and memory-leak fixes so long StaxRip/AvsPmod sessions neither
  grow RAM nor keep VRAM allocated after the preview is closed.

## Quality vs the CPU engine

Measured against the original multi-threaded CPU engine on real 4K clips and
on the 51-fixture matrix:

| Measure | Result |
|---|---|
| Pixels differing from the CPU output | **0.001 %**, always by at most **±1 LSB** (float rounding on exact boundaries) |
| PSNR vs CPU output | **≈ 110 dB** (visually identical) |
| Regression matrix (51 fixtures, all modes/bit depths) | **0 failures** |
| Determinism | identical MD5 across runs, processes and threads |

The only differences come from the GPU's own FFT library rounding slightly
differently than FFTW; they are at the last bit of the format.

## Speed vs the CPU engine

Ryzen 7 5700X + RTX 3050 Laptop 6 GB, warm GPU, milliseconds per 4K frame:

| Preset | CPU | GPU (this plugin) | Speed-up |
|---|---|---|---|
| Light (bt=1) | 242–272 ms | **56 ms** | ~4.5× |
| Medium (bt=3) | 241–271 ms | **51–54 ms** | ~4.9× |
| Heavy (bt=3, wide) | 273–314 ms | **53–60 ms** | ~5.2× |
| bt=5 | 292–315 ms | **57–67 ms** | ~4.9× |

At 1080p the gap is larger: 37.2 ms (CPU) vs **5.7 ms** (GPU) ≈ 6.6×. The
plugin is also ~10–20 % faster than the separate `fft3dcl` port, and supports
`bt=5`, which that port does not.

## Memory

- No leaks: RAM stays flat across repeated previews, and VRAM is released when
  the VapourSynth core is destroyed (verified in StaxRip).
- VRAM footprint at 4K: ~1.5 GiB (luma only) to ~2.9 GiB (all planes, bt=5) —
  fits comfortably on a 6 GB card.
- Uses 2.5–3 GB less system RAM than the `fft3dcl` port at 4K.

## Requirements

- Windows x64, VapourSynth R68 or newer (API 4).
- Any OpenCL 1.2+ GPU with a working `OpenCL.dll` (NVIDIA, AMD, Intel).

## Usage

Put `fft3dneocl.dll` in your VapourSynth plugins directory, then:

```python
import vapoursynth as vs
core = vs.core
clip = core.ffms2.VideoSource("input.mkv")
clip = core.fft3dneocl.FFT3D(clip, bt=3, sigma=2.0, bw=32, bh=32)
clip.set_output()
```

All parameter names, order and defaults are identical to the original
`neo-fft3d` plugin — see [docs/params.md](docs/params.md) (in French).

## Building

```
cmake -S fft3dneocl -B fft3dneocl/build -G "Visual Studio 17 2022" -A x64
cmake --build fft3dneocl/build --config Release
```

The CPU reference harness used for validation builds the same way from
`reference/`. Regression tools live in `tools/` (see the header comment of
each script).

## Repository layout

| Path | Contents |
|---|---|
| `fft3dneocl/` | the OpenCL plugin (sources, kernels, CMake) |
| `src/`, `main.cpp` | the original CPU engine (upstream), plus diagnostic hooks |
| `reference/` | CPU reference harness used to validate the port |
| `tools/` | regression matrix, benchmarks, fixture regeneration |
| `tests/` | fixtures and result sheets for the 51-test matrix and benchmarks |
| `docs/` | design, validation and benchmark notes (French) |
| `third_party/` | FFTW, OpenCL headers, VapourSynth headers |

## Documentation

The detailed notes are in [`docs/`](docs/) (French, kept as written):
porting map, parameter reference, validation status, benchmark reports and
the memory-leak post-mortem.

## License

GPL-2.0, same as the upstream neo_FFT3D project.
