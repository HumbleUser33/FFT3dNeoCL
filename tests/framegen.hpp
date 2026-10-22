// Deterministic synthetic frame generator, shared by the CPU reference
// harness and the GPU test harness so both consume identical pixels.
//
// Content per plane/frame: gradients + a moving block + diagonal texture +
// one impulse + seeded noise. Bit-depth and float aware.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

struct GenFormat {
  // Planes/subsampling identical to DSFormat semantics
  int Planes {3};
  int SSW {0}, SSH {0};
  int BitsPerSample {8};
  int BytesPerSample {1};
  bool IsFloat {false};
};

inline GenFormat GenFormatFromName(const std::string& name, int bitdepth, bool isfloat) {
  GenFormat f;
  if (name == "GRAY") { f.Planes = 1; f.SSW = 0; f.SSH = 0; }
  else if (name == "YUV420") { f.Planes = 3; f.SSW = 1; f.SSH = 1; }
  else if (name == "YUV422") { f.Planes = 3; f.SSW = 1; f.SSH = 0; }
  else if (name == "YUV444") { f.Planes = 3; f.SSW = 0; f.SSH = 0; }
  else if (name == "RGBP")   { f.Planes = 3; f.SSW = 0; f.SSH = 0; }
  else f.Planes = -1;
  f.BitsPerSample = bitdepth;
  f.IsFloat = isfloat;
  f.BytesPerSample = isfloat ? 4 : (bitdepth <= 8 ? 1 : 2);
  return f;
}

inline uint32_t gen_hash(uint32_t x) {
  // splitmix32
  x += 0x9e3779b9u;
  x = (x ^ (x >> 16)) * 0x21f0aaadu;
  x = (x ^ (x >> 15)) * 0x735a2d97u;
  return x ^ (x >> 15);
}

// Generate one plane of one frame. Writes BytesPerSample-sized samples,
// row length = width samples. Values are deterministic for given inputs.
inline void gen_plane(const GenFormat& f, int plane, int frame, int width, int height,
                      uint32_t seed, std::vector<unsigned char>& out) {
  const int maxv = f.IsFloat ? 0 : ((1 << f.BitsPerSample) - 1);
  const double chroma_base = f.IsFloat ? 0.5 : (plane > 0 ? (double)(1 << (f.BitsPerSample - 1)) : 0.0);
  const double full = f.IsFloat ? 1.0 : (double)maxv;

  out.assign((size_t)width * height * f.BytesPerSample, 0);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // base: smooth gradients (low frequencies)
      double v = 0.30 + 0.25 * (double)x / (double)width + 0.15 * (double)y / (double)height;

      // diagonal texture (mid frequencies)
      v += 0.06 * std::sin(0.35 * (x + 2 * y) + 0.7 * frame);

      // moving block (tests temporal modes): 24x24 bright square
      int bx = 40 + 7 * frame, by = 30 + 5 * frame;
      if (x >= bx && x < bx + 24 && y >= by && y < by + 24)
        v += 0.35;

      // second static block with dark interior (edges, halos)
      if (x >= 100 && x < 160 && y >= 80 && y < 120)
        v -= 0.30;

      // impulse (high frequency)
      if (x == 17 && y == 13)
        v += 0.5;

      // seeded noise, stronger on luma
      uint32_t h = gen_hash((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^
                            (uint32_t)frame * 83492791u ^ (uint32_t)plane * 2654435761u ^ seed);
      double noise_amp = (plane == 0) ? 0.05 : 0.02;
      v += noise_amp * ((double)(h & 0xffff) / 65535.0 - 0.5) * 2.0;

      // chroma planes: keep near neutral, gentler content
      if (plane > 0 && !f.IsFloat)
        v = 0.5 + (v - 0.5) * 0.4;

      if (v < 0.0) v = 0.0;
      if (v > 1.0) v = 1.0;

      unsigned char* dst = out.data() + ((size_t)y * width + x) * f.BytesPerSample;
      if (f.IsFloat) {
        float fv = (float)(v * (plane > 0 ? 1.0 : 1.0)); // float: full range 0..1
        std::memcpy(dst, &fv, 4);
      } else if (f.BytesPerSample == 1) {
        int iv = (int)std::lround(v * maxv);
        if (iv < 0) iv = 0; if (iv > maxv) iv = maxv;
        dst[0] = (unsigned char)iv;
      } else {
        int iv = (int)std::lround(v * maxv);
        if (iv < 0) iv = 0; if (iv > maxv) iv = maxv;
        uint16_t sv = (uint16_t)iv;
        std::memcpy(dst, &sv, 2);
      }
      (void)chroma_base;
    }
  }
}
