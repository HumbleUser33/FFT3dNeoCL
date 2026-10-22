// tables_check: validates gpu/tables.hpp (geometry + host tables) against the
// CPU harness stage dumps. Usage: tables_check <dumpdir> [plane]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "../src/gpu/tables.hpp"

static std::vector<unsigned char> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> v(n);
  if (n && fread(v.data(), 1, n, f) != n) { fprintf(stderr, "read error\n"); exit(1); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: tables_check <dumpdir> [plane]\n"); return 1; }
  std::string dir = argv[1];
  int plane = argc > 2 ? atoi(argv[2]) : 0;
  std::string P = "p" + std::to_string(plane);

  // geom from dump
  std::string gs;
  {
    FILE* f = fopen((dir + "/geom_" + P + ".txt").c_str(), "rb");
    if (!f) { fprintf(stderr, "no geom file\n"); return 1; }
    char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) gs.append(buf, n);
    fclose(f);
  }
  auto gv = [&](const char* key) {
    std::string k = std::string("\n") + key + "=";
    size_t p = gs.find(k);
    if (p != std::string::npos) return atoi(gs.c_str() + p + k.size());
    std::string k0 = std::string(key) + "=";
    if (gs.compare(0, k0.size(), k0) == 0) return atoi(gs.c_str() + k0.size());
    return -1;
  };

  // meta (written by the harness next to the dumped input)
  int bits = 8, ssw = 0, ssh = 0, Wf = 64, Hf = 64;
  {
    FILE* f = fopen((dir + "/input.raw.meta.txt").c_str(), "rb");
    char buf[4096] = {}; size_t n = f ? fread(buf, 1, sizeof(buf)-1, f) : 0;
    if (f) fclose(f);
    std::string ms(buf, n);
    auto mv = [&](const char* k) { size_t p = ms.find(k); return p == std::string::npos ? -1 : atoi(ms.c_str() + p + strlen(k)); };
    Wf = mv("width="); Hf = mv("height="); bits = mv("bitdepth=");
    ssw = mv("ssw="); ssh = mv("ssh=");
  }
  int bpp = bits <= 8 ? 1 : (bits <= 16 ? 2 : 4);

  gpu::FilterParams p;
  p.bw = gv("bw"); p.bh = gv("bh");
  p.ow = gv("ow"); p.oh = gv("oh");
  p.wintype = gv("wintype"); if (p.wintype < 0) p.wintype = 0;
  p.interlaced = gv("interlaced"); if (p.interlaced < 0) p.interlaced = 0;
  p.l = gv("l"); p.t = gv("t"); p.r = gv("r"); p.b = gv("b");
  if (p.l < 0) p.l = p.t = p.r = p.b = 0;
  p.bits = bits; p.bpp = bpp;
  p.planeW = Wf; p.planeH = Hf;
  p.isChroma = plane > 0 ? 1 : 0; // fixtures: plane>0 means YUV chroma
  p.xShift = p.isChroma ? ssw : 0;
  p.yShift = p.isChroma ? ssh : 0;
  // defaults used by the fixtures
  p.sigma = 2.0f; p.beta = 1.0f; p.scutoff = 0.3f; p.svr = 1.0f;
  p.smin = 4.0f; p.smax = 20.0f; p.hr = 2.0f; p.ht = 50.0f; p.pcutoff = 0.1f;

  gpu::scale_sigmas(p);
  gpu::Geometry g {};
  if (!gpu::compute_geometry(p, g)) { fprintf(stderr, "geometry FAILED\n"); return 1; }
  gpu::Tables t {};
  gpu::make_tables(p, g, t);

  int fails = 0;
  auto CK = [&](const char* name, int got, int ref) {
    bool ok = got == ref;
    if (!ok) fails++;
    printf("%-14s got=%-8d ref=%-8d %s\n", name, got, ref, ok ? "OK" : "MISMATCH");
  };
  CK("nox", g.nox, gv("nox"));
  CK("noy", g.noy, gv("noy"));
  CK("ow", g.ow, gv("ow"));
  CK("oh", g.oh, gv("oh"));
  CK("coverwidth", g.coverwidth, gv("coverwidth"));
  CK("coverheight", g.coverheight, gv("coverheight"));
  CK("coverpitch", g.coverpitch, gv("coverpitch"));
  CK("insize", g.insize, gv("insize"));
  CK("outwidth", g.outwidth, gv("outwidth"));
  CK("outpitch", g.outpitch, gv("outpitch"));
  CK("outsize", g.outsize, gv("outsize"));
  CK("mirw", g.mirw, gv("mirw"));
  CK("mirh", g.mirh, gv("mirh"));

  auto CMP = [&](const char* name, const std::vector<float>& v) {
    auto ref = read_file(dir + "/" + P + "_f-01_win_" + name + ".bin");
    bool ok = ref.size() == v.size() * 4 &&
              memcmp(ref.data(), v.data(), ref.size()) == 0;
    if (!ok) {
      fails++;
      size_t n = v.size();
      for (size_t i = 0; i < n; i++) {
        float r = 0; if (ref.size() == n * 4) memcpy(&r, &ref[i * 4], 4);
        if (memcmp(&r, &v[i], 4) != 0) {
          printf("%-10s first diff at %zu: got=%.9g ref=%.9g\n", name, i, v[i], r);
          break;
        }
      }
      printf("%-10s MISMATCH (size got=%zu ref=%zu)\n", name, v.size() * 4, ref.size());
    } else {
      printf("%-10s bit-exact\n", name);
    }
  };
  CMP("wanxl", t.wanxl); CMP("wanxr", t.wanxr);
  CMP("wanyl", t.wanyl); CMP("wanyr", t.wanyr);
  CMP("wsynxl", t.wsynxl); CMP("wsynxr", t.wsynxr);
  CMP("wsynyl", t.wsynyl); CMP("wsynyr", t.wsynyr);

  auto SCAL = [&](const char* k, float got) {
    auto v = read_file(dir + "/" + P + "_f-01_" + k + ".txt");
    std::string s((const char*)v.data(), v.size());
    float ref = (float)atof(s.c_str());
    bool ok = memcmp(&got, &ref, 4) == 0;
    if (!ok) fails++;
    printf("%-28s got=%.9g ref=%.9g %s\n", k, got, ref, ok ? "bit-exact" : "MISMATCH");
  };
  SCAL("norm", t.norm);
  SCAL("sigmaSquaredNoiseNormed2D", t.sigmaSquaredNoiseNormed2D);
  SCAL("sigmaSquaredSharpenMinNormed", t.sigmaSquaredSharpenMinNormed);
  SCAL("sigmaSquaredSharpenMaxNormed", t.sigmaSquaredSharpenMaxNormed);
  SCAL("ht2n", t.ht2n);

  auto CMP2 = [&](const char* name, const std::vector<float>& v) {
    auto ref = read_file(dir + "/" + P + "_f-01_" + name + ".bin");
    bool ok = ref.size() == v.size() * 4;
    int shown = 0;
    if (ok) {
      // engine only writes columns < outwidth; padding columns are garbage there
      // and never read by the kernel -> compare the meaningful columns only
      for (size_t i = 0; i < v.size() && shown < 3; i++) {
        if ((int)(i % g.outpitch) >= g.outwidth) continue;
        float r = 0; memcpy(&r, &ref[i * 4], 4);
        if (memcmp(&r, &v[i], 4) != 0) { printf("  %s diff at %zu: got=%.9g ref=%.9g\n", name, i, v[i], r); shown++; }
      }
      ok = shown == 0;
    }
    if (!ok) fails++;
    printf("%-10s %s\n", name, ok ? "bit-exact (meaningful cols)" : "MISMATCH");
  };
  CMP2("wsharpen", t.wsharpen);
  CMP2("wdehalo", t.wdehalo);

  printf("%s\n", fails ? "TABLES CHECK FAILED" : "all tables/geometry bit-exact");
  return fails ? 1 : 0;
}
