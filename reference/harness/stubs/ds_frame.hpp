// Stub of dualsynth DSFrame for the standalone reference harness.
// Holds plain memory buffers instead of VS/AVS frames. The engine only uses:
// FrameWidth/Height, SrcPointers, StrideBytes, DstPointers, Format, Create(bool).
#pragma once

#include <cstdlib>
#include <cstring>

struct DSFrame
{
  int FrameWidth {0}, FrameHeight {0};

  const unsigned char ** SrcPointers {nullptr};
  int * StrideBytes {nullptr};
  unsigned char ** DstPointers {nullptr};
  DSFormat Format;

  // harness storage: when owner, 'storage' holds every plane (height * pitch each)
  unsigned char * storage {nullptr};
  bool owner {false};

  // plane dimensions for plane p
  int plane_width(int p) const { return p == 0 ? FrameWidth : (FrameWidth >> Format.SSW); }
  int plane_height(int p) const { return p == 0 ? FrameHeight : (FrameHeight >> Format.SSH); }

  void free_arrays() {
    if (SrcPointers) { delete[] SrcPointers; SrcPointers = nullptr; }
    if (DstPointers) { delete[] DstPointers; DstPointers = nullptr; }
    if (StrideBytes) { delete[] StrideBytes; StrideBytes = nullptr; }
    if (owner && storage) { std::free(storage); storage = nullptr; }
    owner = false;
  }

  DSFrame() {}
  ~DSFrame() { free_arrays(); }

  // non-owning view over external plane buffers
  DSFrame(const unsigned char* const* ptrs, const int* strides, const DSFormat& fmt, int w, int h)
    : FrameWidth(w), FrameHeight(h), Format(fmt)
  {
    SrcPointers = new const unsigned char*[Format.Planes];
    StrideBytes = new int[Format.Planes];
    for (int i = 0; i < Format.Planes; i++) {
      SrcPointers[i] = ptrs[i];
      StrideBytes[i] = strides[i];
    }
  }

  DSFrame Create(bool copy) const
  {
    DSFrame nf;
    nf.FrameWidth = FrameWidth;
    nf.FrameHeight = FrameHeight;
    nf.Format = Format;

    size_t total = 0;
    int pitches[4];
    for (int p = 0; p < Format.Planes; p++) {
      pitches[p] = StrideBytes ? StrideBytes[p] : plane_width(p) * Format.BytesPerSample;
      total += (size_t)pitches[p] * plane_height(p);
    }
    nf.storage = (unsigned char*)std::malloc(total);
    nf.owner = true;

    nf.SrcPointers = new const unsigned char*[Format.Planes];
    nf.DstPointers = new unsigned char*[Format.Planes];
    nf.StrideBytes = new int[Format.Planes];

    unsigned char* cur = nf.storage;
    size_t left = total;
    for (int p = 0; p < Format.Planes; p++) {
      nf.StrideBytes[p] = pitches[p];
      nf.DstPointers[p] = cur;
      nf.SrcPointers[p] = cur;
      if (copy && SrcPointers)
        std::memcpy(cur, SrcPointers[p], (size_t)pitches[p] * plane_height(p));
      cur += (size_t)pitches[p] * plane_height(p);
      left -= (size_t)pitches[p] * plane_height(p);
    }
    return nf;
  }

  // shallow copy: the original keeps ownership
  DSFrame(const DSFrame& old)
    : FrameWidth(old.FrameWidth), FrameHeight(old.FrameHeight), Format(old.Format)
  {
    if (old.SrcPointers) {
      SrcPointers = new const unsigned char*[Format.Planes];
      std::copy_n(old.SrcPointers, Format.Planes, SrcPointers);
    }
    if (old.DstPointers) {
      DstPointers = new unsigned char*[Format.Planes];
      std::copy_n(old.DstPointers, Format.Planes, DstPointers);
    }
    if (old.StrideBytes) {
      StrideBytes = new int[Format.Planes];
      std::copy_n(old.StrideBytes, Format.Planes, StrideBytes);
    }
  }
  DSFrame& operator=(const DSFrame& old) {
    if (&old == this) return *this;
    free_arrays();
    FrameWidth = old.FrameWidth; FrameHeight = old.FrameHeight; Format = old.Format;
    if (old.SrcPointers) {
      SrcPointers = new const unsigned char*[Format.Planes];
      std::copy_n(old.SrcPointers, Format.Planes, SrcPointers);
    }
    if (old.DstPointers) {
      DstPointers = new unsigned char*[Format.Planes];
      std::copy_n(old.DstPointers, Format.Planes, DstPointers);
    }
    if (old.StrideBytes) {
      StrideBytes = new int[Format.Planes];
      std::copy_n(old.StrideBytes, Format.Planes, StrideBytes);
    }
    return *this;
  }
  DSFrame(DSFrame&& old) noexcept
    : FrameWidth(old.FrameWidth), FrameHeight(old.FrameHeight), Format(old.Format),
      SrcPointers(old.SrcPointers), StrideBytes(old.StrideBytes), DstPointers(old.DstPointers),
      storage(old.storage), owner(old.owner)
  {
    old.SrcPointers = nullptr; old.DstPointers = nullptr; old.StrideBytes = nullptr;
    old.storage = nullptr; old.owner = false;
  }
  DSFrame& operator=(DSFrame&& old) noexcept {
    if (&old == this) return *this;
    free_arrays();
    FrameWidth = old.FrameWidth; FrameHeight = old.FrameHeight; Format = old.Format;
    SrcPointers = old.SrcPointers; StrideBytes = old.StrideBytes; DstPointers = old.DstPointers;
    storage = old.storage; owner = old.owner;
    old.SrcPointers = nullptr; old.DstPointers = nullptr; old.StrideBytes = nullptr;
    old.storage = nullptr; old.owner = false;
    return *this;
  }
};
