# FFT3dNeoCL v1.0 — summary

OpenCL GPU port of the [neo_FFT3D](https://github.com/HomeOfAviSynthPlusEvolution/neo_FFT3D)
FFT denoiser, delivered as a VapourSynth plugin (`fft3dneocl.dll`, API 4).
Same algorithm, same parameters — it just runs on the GPU.

- **Complete port**: blocktype -1 to 5 (Kalman and bm1 included), pattern /
  `pfactor`, sharpen, dehalo, crop, interlaced, window types, plane selection;
  8/10/12/14/16-bit integer and 32-bit float; GRAY, YUV420, YUV422, YUV444.
- **Quality**: identical to the CPU engine — 0.001 % of pixels differ by at
  most ±1 LSB (float rounding), PSNR ≈ 110 dB, 0 failures on the 51-fixture
  regression matrix, byte-identical output across runs.
- **Speed**: 4–5× faster than the CPU engine at 4K (≈ 51–67 ms/frame vs
  240–315 ms), ~6.6× at 1080p; 10–20 % faster than the `fft3dcl` port, and
  supports `bt=5` which `fft3dcl` does not.
- **Memory**: no RAM/VRAM leaks across repeated previews (verified in
  StaxRip); ~1.5–2.9 GiB VRAM at 4K; 2.5–3 GB less RAM than `fft3dcl`.

See [README.md](README.md) for details, benchmarks and build instructions;
in-depth notes (French) in [`docs/`](docs/).
