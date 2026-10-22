// NEOCL-DUMP: stage dump helper for GPU port validation.
// Only compiled when FFT3D_DUMP is defined; never changes numerics.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstddef>

namespace neocl_dump {
  inline const char* dir() {
    static const char* d = getenv("FFT3D_DUMPDIR");
    return d;
  }
  inline void buf(const char* stage, int plane, int frame, const void* p, size_t bytes) {
    const char* d = dir();
    if (!d || !p) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/p%d_f%03d_%s.bin", d, plane, frame, stage);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(p, 1, bytes, f); fclose(f); }
  }
  inline void scal(const char* stage, int plane, int frame, double v) {
    const char* d = dir();
    if (!d) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/p%d_f%03d_%s.txt", d, plane, frame, stage);
    FILE* f = fopen(path, "wb");
    if (f) { fprintf(f, "%.17g\n", v); fclose(f); }
  }
}
#define DUMP_BUF(stage, plane, frame, ptr, bytes) neocl_dump::buf(stage, plane, frame, ptr, bytes)
#define DUMP_SCAL(stage, plane, frame, v) neocl_dump::scal(stage, plane, frame, (double)(v))
