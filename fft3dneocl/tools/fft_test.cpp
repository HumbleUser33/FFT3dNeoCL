// GPU FFT unit test: runs fft_fwd / fft_inv on CPU stage dumps and compares.
// Usage: fft_test <dumpdir> [bw bh]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "../src/ocl/ocl_loader.hpp"
#include "../src/ocl/context.hpp"

struct cf { float x, y; };

static std::vector<float> twiddle_table(int N) {
  // per stage ncur = N, N/2, ..., 2: entries exp(-2 pi i p / ncur), p < ncur/2
  // packed at offset N - ncur (total N-1 complex = 2(N-1) floats)
  std::vector<float> t(2 * (N - 1));
  int off = 0;
  for (int ncur = N; ncur >= 2; ncur >>= 1) {
    for (int p = 0; p < ncur / 2; p++) {
      double a = -2.0 * 3.14159265358979323846 * p / ncur;
      t[2 * (off + p)] = (float)std::cos(a);
      t[2 * (off + p) + 1] = (float)std::sin(a);
    }
    off += ncur / 2;
  }
  return t;
}

static std::vector<unsigned char> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> d(n);
  fread(d.data(), 1, n, f);
  fclose(f);
  return d;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "tests/dump_bt1";
  int plane = 0;
  for (int i = 2; i < argc; i++)
    if (!strncmp(argv[i], "plane=", 6)) plane = atoi(argv[i] + 6);
  std::string P = "p" + std::to_string(plane);

  // geometry
  auto geom = read_file(dir + "/geom_" + P + ".txt");
  std::string gs((char*)geom.data(), geom.size());
  auto gv = [&](const char* key) {
    // keys are one per line; anchor at line start so "t=" does not match "bt="
    std::string k = std::string("\n") + key + "=";
    size_t p = gs.find(k);
    if (p != std::string::npos) return atoi(gs.c_str() + p + k.size());
    std::string k0 = std::string(key) + "=";
    if (gs.compare(0, k0.size(), k0) == 0) return atoi(gs.c_str() + k0.size());
    return -1;
  };
  int nox = gv("nox"), noy = gv("noy"), outwidth = gv("outwidth"), outpitch = gv("outpitch");
  int insize = gv("insize"), outsize = gv("outsize"), bw = gv("bw"), bh = gv("bh");
  int ow = gv("ow"), oh = gv("oh");
  int coverwidth = gv("coverwidth"), coverheight = gv("coverheight"), coverpitch = gv("coverpitch");
  int mirw = gv("mirw"), mirh = gv("mirh");

  int Wf = 64, Hf = 64, bitdepth = 8, bpp = 1, ssw = 0, ssh = 0;
  {
    auto meta = read_file(dir + "/input.raw.meta.txt");
    std::string ms((char*)meta.data(), meta.size());
    size_t p = ms.find("width="); Wf = atoi(ms.c_str() + p + 6);
    p = ms.find("height="); Hf = atoi(ms.c_str() + p + 7);
    p = ms.find("bitdepth="); bitdepth = atoi(ms.c_str() + p + 9);
    p = ms.find("bytespersample="); bpp = atoi(ms.c_str() + p + 15);
    p = ms.find("ssw="); ssw = atoi(ms.c_str() + p + 4);
    p = ms.find("ssh="); ssh = atoi(ms.c_str() + p + 4);
  }
  // plane dimensions and byte offset inside input.raw (planes concatenated)
  int planeW = plane == 0 ? Wf : Wf >> ssw;
  int planeH = plane == 0 ? Hf : Hf >> ssh;
  size_t frame_off = 0;
  for (int p = 0; p < plane; p++)
    frame_off += (size_t)(p == 0 ? Wf : Wf >> ssw) * (p == 0 ? Hf : Hf >> ssh) * bpp;
  int interlaced = gv("interlaced");
  float planeBase = (float)gv("planeBase");
  // crop (engine l/t/r/b, shifted for chroma; absent in old dumps -> 0)
  int cl = gv("l"), ct = gv("t"), cr = gv("r"), cb = gv("b");
  if (cl < 0) cl = ct = cr = cb = 0;
  if (plane > 0) { cl >>= ssw; cr >>= ssw; ct >>= ssh; cb >>= ssh; }
  int pw = planeW - cl - cr;
  int ph = planeH - ct - cb;
  size_t crop_off = frame_off + ((size_t)ct * planeW + cl) * bpp;
  printf("geom: nox=%d noy=%d outwidth=%d outpitch=%d insize=%d outsize=%d bw=%d bh=%d bitdepth=%d bpp=%d plane=%d %dx%d crop=%d,%d,%d,%d interlaced=%d planeBase=%.0f\n",
         nox, noy, outwidth, outpitch, insize, outsize, bw, bh, bitdepth, bpp, plane, pw, ph, cl, ct, cr, cb, interlaced, planeBase);

  auto inb = read_file(dir + "/" + P + "_f000_in.bin");
  auto fftb = read_file(dir + "/" + P + "_f000_fft.bin");
  auto appb = read_file(dir + "/" + P + "_f000_applied.bin");
  auto invb = read_file(dir + "/" + P + "_f000_inv.bin");

  // kernel source
  auto kb = read_file("fft3dneocl/src/kernels/fft3d.cl");
  std::string src((char*)kb.data(), kb.size());

  char err[4096];
  ocl::Api api;
  // context create loads api; list devices first with a temp api
  if (!api.load(err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
  auto devs = ocl::Context::list_devices(api, false);
  for (size_t i = 0; i < devs.size(); i++)
    printf("gpu %zu: %s (%s)\n", i, devs[i].device_name.c_str(), devs[i].version.c_str());

  // pick FFT path: fused (one work-group per block) if local memory suffices
  cl_ulong need_local = (cl_ulong)2 * bh * (bw + 1) * 8; // A + B, float2, pitch bw+1
  int fused = (devs[0].local_mem >= need_local) ? 1 : 0;
  if (argc > 2 && !strcmp(argv[2], "split")) fused = 0; // force split path for debugging
  printf("fft path: %s (need %llu, device %llu)\n", fused ? "fused" : "split",
         (unsigned long long)need_local, (unsigned long long)devs[0].local_mem);

  char bopts[128];
  snprintf(bopts, sizeof(bopts), "-cl-std=CL1.2 -DBW=%d -DBH=%d -DOW=%d -DOH=%d -DBPP=%d -DFFT_FUSED=%d",
           bw, bh, gv("ow"), gv("oh"), bpp, fused);

  // fresh api inside context
  ocl::Context ctx;
  if (!ctx.create(src, bopts, 0, -1, err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
  printf("device: %s local_mem=%llu\n", ctx.info.device_name.c_str(), (unsigned long long)ctx.info.local_mem);

  int nblocks = nox * noy;
  cl_int e;

  auto twR = twiddle_table(bw);
  auto twC = twiddle_table(bh);

  cl_mem d_in = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inb.size(), inb.data(), &e);
  cl_mem d_out = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)outsize * 8, nullptr, &e);
  cl_mem d_twR = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, twR.size() * 4, twR.data(), &e);
  cl_mem d_twC = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, twC.size() * 4, twC.data(), &e);
  cl_mem d_tmp = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)insize * 8, nullptr, &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "buffer: %s\n", ocl::err_str(e).c_str()); return 1; }

  // FFT dispatch: fused single-kernel or split rows+cols
  cl_kernel k = nullptr, ki = nullptr, kfr = nullptr, kfc2 = nullptr, kir = nullptr, kic = nullptr;
  if (fused) {
    k = ctx.kernel("fft_fwd", &e);
    if (e != CL_SUCCESS) { fprintf(stderr, "kernel fft_fwd: %s\n", ocl::err_str(e).c_str()); return 1; }
    ki = ctx.kernel("fft_inv", &e);
    if (e != CL_SUCCESS) { fprintf(stderr, "kernel fft_inv: %s\n", ocl::err_str(e).c_str()); return 1; }
  } else {
    kfr = ctx.kernel("fft_fwd_rows", &e);
    kfc2 = ctx.kernel("fft_fwd_cols", &e);
    kir = ctx.kernel("fft_inv_rows", &e);
    kic = ctx.kernel("fft_inv_cols", &e);
    if (e != CL_SUCCESS) { fprintf(stderr, "split fft kernels: %s\n", ocl::err_str(e).c_str()); return 1; }
  }

  size_t wg = 256;
  size_t gsize[1] = { (size_t)nblocks * wg };
  size_t lsize[1] = { wg };

  auto run_fwd = [&](cl_mem src, cl_mem dst) {
    if (fused) {
      api.SetKernelArg(k, 0, sizeof(cl_mem), &src);
      api.SetKernelArg(k, 1, sizeof(cl_mem), &dst);
      api.SetKernelArg(k, 2, sizeof(cl_mem), &d_twR);
      api.SetKernelArg(k, 3, sizeof(cl_mem), &d_twC);
      api.SetKernelArg(k, 4, sizeof(int), &outpitch);
      api.EnqueueNDRangeKernel(ctx.queue, k, 1, nullptr, gsize, lsize, 0, nullptr, nullptr);
    } else {
      api.SetKernelArg(kfr, 0, sizeof(cl_mem), &src);
      api.SetKernelArg(kfr, 1, sizeof(cl_mem), &d_tmp);
      api.SetKernelArg(kfr, 2, sizeof(cl_mem), &d_twR);
      size_t g[1] = { (size_t)nblocks * bh * bw }, l[1] = { (size_t)bw };
      cl_int e1 = api.EnqueueNDRangeKernel(ctx.queue, kfr, 1, nullptr, g, l, 0, nullptr, nullptr);
      if (e1 != CL_SUCCESS) printf("  fft_fwd_rows enqueue: %s\n", ocl::err_str(e1).c_str());
      api.SetKernelArg(kfc2, 0, sizeof(cl_mem), &d_tmp);
      api.SetKernelArg(kfc2, 1, sizeof(cl_mem), &dst);
      api.SetKernelArg(kfc2, 2, sizeof(cl_mem), &d_twC);
      api.SetKernelArg(kfc2, 3, sizeof(int), &outpitch);
      size_t g2[1] = { (size_t)nblocks * (bw / 2 + 1) * bh }, l2[1] = { (size_t)bh };
      cl_int e2 = api.EnqueueNDRangeKernel(ctx.queue, kfc2, 1, nullptr, g2, l2, 0, nullptr, nullptr);
      if (e2 != CL_SUCCESS) printf("  fft_fwd_cols enqueue: %s\n", ocl::err_str(e2).c_str());
    }
    api.Finish(ctx.queue);
  };
  auto run_inv = [&](cl_mem spec, cl_mem real) {
    if (fused) {
      api.SetKernelArg(ki, 0, sizeof(cl_mem), &spec);
      api.SetKernelArg(ki, 1, sizeof(cl_mem), &real);
      api.SetKernelArg(ki, 2, sizeof(cl_mem), &d_twR);
      api.SetKernelArg(ki, 3, sizeof(cl_mem), &d_twC);
      api.SetKernelArg(ki, 4, sizeof(int), &outpitch);
      api.EnqueueNDRangeKernel(ctx.queue, ki, 1, nullptr, gsize, lsize, 0, nullptr, nullptr);
    } else {
      api.SetKernelArg(kir, 0, sizeof(cl_mem), &spec);
      api.SetKernelArg(kir, 1, sizeof(cl_mem), &d_tmp);
      api.SetKernelArg(kir, 2, sizeof(cl_mem), &d_twR);
      api.SetKernelArg(kir, 3, sizeof(int), &outpitch);
      size_t g[1] = { (size_t)nblocks * bh * bw }, l[1] = { (size_t)bw };
      api.EnqueueNDRangeKernel(ctx.queue, kir, 1, nullptr, g, l, 0, nullptr, nullptr);
      api.SetKernelArg(kic, 0, sizeof(cl_mem), &d_tmp);
      api.SetKernelArg(kic, 1, sizeof(cl_mem), &real);
      api.SetKernelArg(kic, 2, sizeof(cl_mem), &d_twC);
      size_t g2[1] = { (size_t)nblocks * bw * bh }, l2[1] = { (size_t)bh };
      api.EnqueueNDRangeKernel(ctx.queue, kic, 1, nullptr, g2, l2, 0, nullptr, nullptr);
    }
    api.Finish(ctx.queue);
  };

  run_fwd(d_in, d_out);

  // debug: dump GPU tmp (row-pass intermediate) for split path
  if (!fused) {
    std::vector<float> tmprows((size_t)insize * 2);
    api.EnqueueReadBuffer(ctx.queue, d_tmp, CL_TRUE, 0, tmprows.size() * 4, tmprows.data(), 0, nullptr, nullptr);
    FILE* f = fopen((dir + "/tmp_rows.bin").c_str(), "wb");
    fwrite(tmprows.data(), 4, tmprows.size(), f); fclose(f);
  }

  std::vector<float> got((size_t)outsize * 2);
  api.EnqueueReadBuffer(ctx.queue, d_out, CL_TRUE, 0, got.size() * 4, got.data(), 0, nullptr, nullptr);

  // compare vs FFTW (only the outwidth used columns per row; padding is garbage)
  const float* ref = (const float*)fftb.data();
  double maxabs = 0, maxrel = 0, sum2 = 0;
  long n = 0;
  for (int b = 0; b < nblocks; b++)
    for (int h = 0; h < bh; h++)
      for (int w = 0; w < outwidth; w++) {
        size_t idx = ((size_t)b * outpitch + h) * outpitch + w;
        double d = std::fabs(got[idx * 2] - ref[idx * 2]);
        double d2 = std::fabs(got[idx * 2 + 1] - ref[idx * 2 + 1]);
        if (d > maxabs) maxabs = d;
        if (d2 > maxabs) maxabs = d2;
        double r = (d > d2 ? d : d2) / (std::fabs(ref[idx * 2]) + std::fabs(ref[idx * 2 + 1]) + 1e-3);
        if (r > maxrel) maxrel = r;
        sum2 += d * d + d2 * d2;
        n += 2;
      }
  printf("fft_fwd vs FFTW: maxabs=%.3e maxrel=%.3e rms=%.3e\n", maxabs, maxrel, std::sqrt(sum2 / n));

  // inverse: run fft_inv on the CPU APPLIED spectrum, compare to CPU inverse
  cl_mem d_fft = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, appb.size(), appb.data(), &e);
  cl_mem d_rinv = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)insize * 4, nullptr, &e);
  run_inv(d_fft, d_rinv);

  std::vector<float> gotinv((size_t)insize);
  api.EnqueueReadBuffer(ctx.queue, d_rinv, CL_TRUE, 0, gotinv.size() * 4, gotinv.data(), 0, nullptr, nullptr);
  const float* refi = (const float*)invb.data();
  maxabs = 0; maxrel = 0; sum2 = 0;
  for (long i = 0; i < insize; i++) {
    double d = std::fabs(gotinv[i] - refi[i]);
    double r = d / (std::fabs(refi[i]) + 1e-3);
    if (d > maxabs) maxabs = d;
    if (r > maxrel) maxrel = r;
    sum2 += d * d;
  }
  printf("fft_inv vs FFTW: maxabs=%.3e maxrel=%.3e rms=%.3e\n", maxabs, maxrel, std::sqrt(sum2 / insize));

  // round-trip: GPU inv of GPU fwd should reproduce the input (x bw*bh, since
  // neither fwd nor inv normalizes; FFTW convention: norm applied later)
  cl_mem d_rt = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)insize * 4, nullptr, &e);
  run_inv(d_out, d_rt); // GPU fwd result still in d_out
  std::vector<float> gotrt((size_t)insize);
  api.EnqueueReadBuffer(ctx.queue, d_rt, CL_TRUE, 0, gotrt.size() * 4, gotrt.data(), 0, nullptr, nullptr);
  const float* srcin = (const float*)inb.data();
  double scale = (double)bw * bh;
  maxabs = 0; sum2 = 0;
  for (long i = 0; i < insize; i++) {
    double d = std::fabs(gotrt[i] / scale - srcin[i]);
    if (d > maxabs) maxabs = d;
    sum2 += d * d;
  }
  printf("round-trip inv(fwd)/N2 vs in: maxabs=%.3e rms=%.3e\n", maxabs, std::sqrt(sum2 / insize));
  api.ReleaseMemObject(d_rt);

  // determinism: run fft_fwd again into a second buffer, compare bit-exactly
  cl_mem d_out2 = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)outsize * 8, nullptr, &e);
  std::vector<float> got2((size_t)outsize * 2);
  auto cmp_used = [&](const std::vector<float>& a, const std::vector<float>& b, const char* label) {
    bool exact = true; long nd = 0; double mx = 0;
    for (int bl = 0; bl < nblocks; bl++)
      for (int h = 0; h < bh; h++)
        for (int w = 0; w < outwidth; w++) {
          size_t idx = (((size_t)bl * outpitch + h) * outpitch + w) * 2;
          if (a[idx] != b[idx] || a[idx + 1] != b[idx + 1]) {
            exact = false; nd++;
            double d = std::fabs((double)a[idx] - b[idx]);
            double d2 = std::fabs((double)a[idx + 1] - b[idx + 1]);
            if (d > mx) mx = d;
            if (d2 > mx) mx = d2;
          }
        }
    printf("determinism (%s): %s  ndiff=%ld maxdiff=%.3e\n", label, exact ? "OK" : "MISMATCH", nd, mx);
  };
  run_fwd(d_in, d_out2);
  api.EnqueueReadBuffer(ctx.queue, d_out2, CL_TRUE, 0, got2.size() * 4, got2.data(), 0, nullptr, nullptr);
  cmp_used(got, got2, "run 2");
  run_fwd(d_in, d_out2);
  api.EnqueueReadBuffer(ctx.queue, d_out2, CL_TRUE, 0, got2.size() * 4, got2.data(), 0, nullptr, nullptr);
  cmp_used(got, got2, "run 3");
  api.ReleaseMemObject(d_out2);

  // ================= buffer stages =================
  auto win = [&](const char* name) { return read_file(dir + "/" + P + "_f-01_win_" + name + ".bin"); };
  auto wanxl = win("wanxl"), wanxr = win("wanxr"), wanyl = win("wanyl"), wanyr = win("wanyr");
  auto wsynxl = win("wsynxl"), wsynxr = win("wsynxr"), wsynyl = win("wsynyl"), wsynyr = win("wsynyr");
  auto normf = read_file(dir + "/" + P + "_f-01_norm.txt");
  float norm = (float)atof((const char*)normf.data());
  auto coverdump = read_file(dir + "/" + P + "_f000_cover.bin");
  auto coverout = read_file(dir + "/" + P + "_f000_coverout.bin");
  auto framein = read_file(dir + "/input.raw");

  size_t g2[2] = { (size_t)coverwidth, (size_t)coverheight };

  // --- frame_to_cover: expect bit-exact ---
  cl_mem d_frame = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)ph * planeW * bpp, framein.data() + crop_off, &e);
  cl_mem d_cover = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)coverheight * coverpitch * bpp, nullptr, &e);
  cl_kernel kfc = ctx.kernel("frame_to_cover", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "kernel frame_to_cover: %s\n", ocl::err_str(e).c_str()); return 1; }
  int src_pitch = planeW * bpp;
  api.SetKernelArg(kfc, 0, sizeof(cl_mem), &d_frame);
  api.SetKernelArg(kfc, 1, sizeof(int), &src_pitch);
  api.SetKernelArg(kfc, 2, sizeof(int), &pw);
  api.SetKernelArg(kfc, 3, sizeof(int), &ph);
  api.SetKernelArg(kfc, 4, sizeof(cl_mem), &d_cover);
  api.SetKernelArg(kfc, 5, sizeof(int), &coverpitch);
  api.SetKernelArg(kfc, 6, sizeof(int), &coverwidth);
  api.SetKernelArg(kfc, 7, sizeof(int), &coverheight);
  api.SetKernelArg(kfc, 8, sizeof(int), &mirw);
  api.SetKernelArg(kfc, 9, sizeof(int), &mirh);
  api.SetKernelArg(kfc, 10, sizeof(int), &interlaced);
  api.EnqueueNDRangeKernel(ctx.queue, kfc, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
  api.Finish(ctx.queue);
  std::vector<unsigned char> gotcov((size_t)coverheight * coverpitch * bpp);
  api.EnqueueReadBuffer(ctx.queue, d_cover, CL_TRUE, 0, gotcov.size(), gotcov.data(), 0, nullptr, nullptr);
  long covdiff = 0;
  for (int y = 0; y < coverheight; y++)
    for (int x = 0; x < coverwidth * bpp; x++) {
      size_t i = (size_t)y * coverpitch * bpp + x;
      if (gotcov[i] != coverdump[i]) {
        if (covdiff < 12)
          printf("  f2c diff at (x=%d,y=%d) got=%d ref=%d\n", x, y, gotcov[i], coverdump[i]);
        covdiff++;
      }
    }
  printf("frame_to_cover vs CPU: diff=%ld / %d (expect 0)\n", covdiff, coverwidth * coverheight * bpp);

  // --- cover_to_overlap: float compare vs CPU 'in' ---
  cl_mem d_covercpu = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, coverdump.size(), coverdump.data(), &e);
  cl_mem d_in2 = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)insize * 4, nullptr, &e);
  cl_mem d_wanxl = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wanxl.size(), wanxl.data(), &e);
  cl_mem d_wanxr = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wanxr.size(), wanxr.data(), &e);
  cl_mem d_wanyl = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wanyl.size(), wanyl.data(), &e);
  cl_mem d_wanyr = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wanyr.size(), wanyr.data(), &e);
  cl_kernel kco = ctx.kernel("cover_to_overlap", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "kernel cover_to_overlap: %s\n", ocl::err_str(e).c_str()); return 1; }
  api.SetKernelArg(kco, 0, sizeof(cl_mem), &d_covercpu);
  api.SetKernelArg(kco, 1, sizeof(int), &coverpitch);
  api.SetKernelArg(kco, 2, sizeof(cl_mem), &d_in2);
  api.SetKernelArg(kco, 3, sizeof(cl_mem), &d_wanxl);
  api.SetKernelArg(kco, 4, sizeof(cl_mem), &d_wanxr);
  api.SetKernelArg(kco, 5, sizeof(cl_mem), &d_wanyl);
  api.SetKernelArg(kco, 6, sizeof(cl_mem), &d_wanyr);
  api.SetKernelArg(kco, 7, sizeof(int), &coverwidth);
  api.SetKernelArg(kco, 8, sizeof(int), &coverheight);
  api.SetKernelArg(kco, 9, sizeof(int), &nox);
  api.SetKernelArg(kco, 10, sizeof(int), &noy);
  api.SetKernelArg(kco, 11, sizeof(float), &planeBase);
  api.EnqueueNDRangeKernel(ctx.queue, kco, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
  api.Finish(ctx.queue);
  std::vector<float> gotin(insize);
  api.EnqueueReadBuffer(ctx.queue, d_in2, CL_TRUE, 0, gotin.size() * 4, gotin.data(), 0, nullptr, nullptr);
  {
    const float* refin = (const float*)inb.data();
    double mx = 0; long nd = 0;
    for (int i = 0; i < insize; i++) {
      double d = std::fabs(gotin[i] - refin[i]);
      if (d > 1e-6) nd++;
      if (d > mx) mx = d;
    }
    printf("cover_to_overlap vs CPU: maxabs=%.3e ndiff=%.1f%%\n", mx, 100.0 * nd / insize);
  }

  // --- overlap_to_cover: byte compare vs CPU 'coverout' ---
  cl_mem d_invcpu = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, invb.size(), invb.data(), &e);
  cl_mem d_cov2 = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)coverheight * coverpitch * bpp, nullptr, &e);
  cl_mem d_wsynxl = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wsynxl.size(), wsynxl.data(), &e);
  cl_mem d_wsynxr = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wsynxr.size(), wsynxr.data(), &e);
  cl_mem d_wsynyl = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wsynyl.size(), wsynyl.data(), &e);
  cl_mem d_wsynyr = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wsynyr.size(), wsynyr.data(), &e);
  cl_kernel koc = ctx.kernel("overlap_to_cover", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "kernel overlap_to_cover: %s\n", ocl::err_str(e).c_str()); return 1; }
  float rounder = bpp == 4 ? 0.0f : 0.5f;
  float maxval = bpp == 4 ? 1.0f : (float)((1u << bitdepth) - 1u);
  api.SetKernelArg(koc, 0, sizeof(cl_mem), &d_invcpu);
  api.SetKernelArg(koc, 1, sizeof(cl_mem), &d_cov2);
  api.SetKernelArg(koc, 2, sizeof(int), &coverpitch);
  api.SetKernelArg(koc, 3, sizeof(cl_mem), &d_wsynxl);
  api.SetKernelArg(koc, 4, sizeof(cl_mem), &d_wsynxr);
  api.SetKernelArg(koc, 5, sizeof(cl_mem), &d_wsynyl);
  api.SetKernelArg(koc, 6, sizeof(cl_mem), &d_wsynyr);
  api.SetKernelArg(koc, 7, sizeof(int), &coverwidth);
  api.SetKernelArg(koc, 8, sizeof(int), &coverheight);
  api.SetKernelArg(koc, 9, sizeof(int), &nox);
  api.SetKernelArg(koc, 10, sizeof(int), &noy);
  api.SetKernelArg(koc, 11, sizeof(float), &norm);
  api.SetKernelArg(koc, 12, sizeof(float), &rounder);
  api.SetKernelArg(koc, 13, sizeof(float), &planeBase);
  api.SetKernelArg(koc, 14, sizeof(float), &maxval);
  api.EnqueueNDRangeKernel(ctx.queue, koc, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
  api.Finish(ctx.queue);
  std::vector<unsigned char> gotcov2((size_t)coverheight * coverpitch * bpp);
  api.EnqueueReadBuffer(ctx.queue, d_cov2, CL_TRUE, 0, gotcov2.size(), gotcov2.data(), 0, nullptr, nullptr);
  {
    long nd = 0; int mx = 0;
    for (int y = 0; y < coverheight; y++)
      for (int x = 0; x < coverwidth * bpp; x++) {
        size_t i = (size_t)y * coverpitch * bpp + x;
        int d = std::abs((int)gotcov2[i] - (int)coverout[i]);
        if (d > 0) {
          if (nd < 12) printf("  o2c diff at (x=%d,y=%d) got=%d ref=%d\n", x, y, gotcov2[i], coverout[i]);
          nd++;
        }
        if (d > mx) mx = d;
      }
    printf("overlap_to_cover vs CPU: ndiff=%ld (%.2f%%) maxabs=%d\n", nd, 100.0 * nd / (coverwidth * coverheight * bpp), mx);
  }

  // --- apply2d: float compare vs CPU 'applied' (on the CPU fft input) ---
  auto gsamp = read_file(dir + "/" + P + "_f-01_gridsample.bin");
  auto wshp = read_file(dir + "/" + P + "_f-01_wsharpen.bin");
  auto wdeh = read_file(dir + "/" + P + "_f-01_wdehalo.bin");
  auto pat2 = read_file(dir + "/" + P + "_f-01_pattern2d.bin");
  auto rdscal = [&](const char* k) {
    auto v = read_file(dir + "/" + P + "_f-01_" + k + ".txt");
    return (float)atof((const char*)v.data());
  };
  float sigma2N = rdscal("sigmaSquaredNoiseNormed2D");
  float minN = rdscal("sigmaSquaredSharpenMinNormed");
  float maxN = rdscal("sigmaSquaredSharpenMaxNormed");
  float ht2n = rdscal("ht2n");
  float degrid = 1.0f, sharpen = 0.0f, dehalo = 0.0f, beta = 1.0f; // dump config defaults
  float lowlimit = (beta - 1.0f) / beta;
  int odist = outpitch * bh;
  int use_pattern = 0;

  cl_mem d_gs = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gsamp.size(), gsamp.data(), &e);
  cl_mem d_wsh = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wshp.size(), wshp.data(), &e);
  cl_mem d_wde = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wdeh.size(), wdeh.data(), &e);
  cl_mem d_pat = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pat2.size(), pat2.data(), &e);
  cl_mem d_gf = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)nblocks * 4, nullptr, &e);
  cl_mem d_app = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)outsize * 8, nullptr, &e);

  cl_mem d_fft2 = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fftb.size(), fftb.data(), &e);
  cl_kernel kgf = ctx.kernel("gridfraction_k", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "kernel gridfraction_k: %s\n", ocl::err_str(e).c_str()); return 1; }
  api.SetKernelArg(kgf, 0, sizeof(cl_mem), &d_fft2); // CPU fft dump
  api.SetKernelArg(kgf, 1, sizeof(cl_mem), &d_gs);
  api.SetKernelArg(kgf, 2, sizeof(float), &degrid);
  api.SetKernelArg(kgf, 3, sizeof(int), &odist);
  api.SetKernelArg(kgf, 4, sizeof(cl_mem), &d_gf);
  size_t g1[1] = { (size_t)nblocks };
  api.EnqueueNDRangeKernel(ctx.queue, kgf, 1, nullptr, g1, nullptr, 0, nullptr, nullptr);

  // copy CPU fft into d_app (apply works in-place)
  api.EnqueueCopyBuffer(ctx.queue, d_fft2, d_app, 0, 0, (size_t)outsize * 8, 0, nullptr, nullptr);

  cl_kernel kap = ctx.kernel("apply2d", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "kernel apply2d: %s\n", ocl::err_str(e).c_str()); return 1; }
  api.SetKernelArg(kap, 0, sizeof(cl_mem), &d_app);
  api.SetKernelArg(kap, 1, sizeof(cl_mem), &d_gf);
  api.SetKernelArg(kap, 2, sizeof(cl_mem), &d_pat);
  api.SetKernelArg(kap, 3, sizeof(cl_mem), &d_gs);
  api.SetKernelArg(kap, 4, sizeof(cl_mem), &d_wsh);
  api.SetKernelArg(kap, 5, sizeof(cl_mem), &d_wde);
  api.SetKernelArg(kap, 6, sizeof(int), &outpitch);
  api.SetKernelArg(kap, 7, sizeof(int), &outwidth);
  api.SetKernelArg(kap, 8, sizeof(int), &bh);
  api.SetKernelArg(kap, 9, sizeof(float), &sigma2N);
  api.SetKernelArg(kap, 10, sizeof(float), &lowlimit);
  api.SetKernelArg(kap, 11, sizeof(float), &sharpen);
  api.SetKernelArg(kap, 12, sizeof(float), &dehalo);
  api.SetKernelArg(kap, 13, sizeof(float), &minN);
  api.SetKernelArg(kap, 14, sizeof(float), &maxN);
  api.SetKernelArg(kap, 15, sizeof(float), &ht2n);
  api.SetKernelArg(kap, 16, sizeof(int), &use_pattern);
  size_t gapp[1] = { (size_t)nblocks * bh * outwidth };
  api.EnqueueNDRangeKernel(ctx.queue, kap, 1, nullptr, gapp, nullptr, 0, nullptr, nullptr);
  api.Finish(ctx.queue);

  std::vector<float> gotapp((size_t)outsize * 2);
  api.EnqueueReadBuffer(ctx.queue, d_app, CL_TRUE, 0, gotapp.size() * 4, gotapp.data(), 0, nullptr, nullptr);
  {
    const float* refap = (const float*)appb.data();
    double mx = 0, rms = 0; long nd = 0, n = 0;
    int wb = 0, wh = 0, ww = 0;
    for (int bl = 0; bl < nblocks; bl++)
      for (int h = 0; h < bh; h++)
        for (int w = 0; w < outwidth; w++) {
          size_t i2 = (((size_t)bl * outpitch + h) * outpitch + w) * 2;
          double d0 = std::fabs(gotapp[i2] - refap[i2]);
          double d1 = std::fabs(gotapp[i2 + 1] - refap[i2 + 1]);
          double d = d0 > d1 ? d0 : d1;
          if (d > 1e-4) nd++;
          if (d > mx) { mx = d; wb = bl; wh = h; ww = w; }
          rms += d0 * d0 + d1 * d1; n += 2;
        }
    printf("apply2d vs CPU: maxabs=%.3e rms=%.3e ndiff(>1e-4)=%ld\n", mx, std::sqrt(rms / n), nd);
    size_t i2 = (((size_t)wb * outpitch + wh) * outpitch + ww) * 2;
    printf("  worst bin b=%d h=%d w=%d got=(%.4f,%.4f) ref=(%.4f,%.4f)\n", wb, wh, ww,
           gotapp[i2], gotapp[i2 + 1], refap[i2], refap[i2 + 1]);
    // gf diagnostics
    std::vector<float> gotgf(nblocks);
    api.EnqueueReadBuffer(ctx.queue, d_gf, CL_TRUE, 0, gotgf.size() * 4, gotgf.data(), 0, nullptr, nullptr);
    printf("  gf[0..3]: %.6f %.6f %.6f %.6f\n", gotgf[0], gotgf[1], gotgf[2], gotgf[3]);
  }

  // ================= end-to-end GPU pipeline vs out.raw =================
  {
    auto outraw = read_file(dir + "/out.raw");
    const unsigned char* outr = outraw.data() + frame_off;
    cl_mem e_frame = api.CreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)ph * planeW * bpp, framein.data() + crop_off, &e);
    cl_mem e_cover = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)coverheight * coverpitch * bpp, nullptr, &e);
    cl_mem e_in = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)insize * 4, nullptr, &e);
    cl_mem e_spec = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)outsize * 8, nullptr, &e);
    cl_mem e_gf = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)nblocks * 4, nullptr, &e);
    cl_mem e_rinv = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)insize * 4, nullptr, &e);
    cl_mem e_cover2 = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)coverheight * coverpitch * bpp, nullptr, &e);
    cl_mem e_dst = api.CreateBuffer(ctx.ctx, CL_MEM_READ_WRITE, (size_t)pw * ph * bpp, nullptr, &e);

    // 1. frame_to_cover
    api.SetKernelArg(kfc, 0, sizeof(cl_mem), &e_frame);
    api.SetKernelArg(kfc, 4, sizeof(cl_mem), &e_cover);
    api.EnqueueNDRangeKernel(ctx.queue, kfc, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
    // 2. cover_to_overlap
    api.SetKernelArg(kco, 0, sizeof(cl_mem), &e_cover);
    api.SetKernelArg(kco, 2, sizeof(cl_mem), &e_in);
    api.EnqueueNDRangeKernel(ctx.queue, kco, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
    // debug: compare e_in with CPU 'in' dump
    {
      std::vector<float> din(insize);
      api.EnqueueReadBuffer(ctx.queue, e_in, CL_TRUE, 0, din.size() * 4, din.data(), 0, nullptr, nullptr);
      const float* rin = (const float*)inb.data();
      double mx = 0; long nd2 = 0; int wx = 0, wy = 0, wb = 0;
      for (int b = 0; b < nblocks; b++)
        for (int i = 0; i < bw * bh; i++) {
          double d = std::fabs(din[(size_t)b * bw * bh + i] - rin[(size_t)b * bw * bh + i]);
          if (d > 1e-4) { nd2++; if (d > mx) { mx = d; wb = b; wy = i / bw; wx = i % bw; } }
        }
      printf("  [dbg] e_in vs CPU in: maxabs=%.3e ndiff=%ld worst b=%d (x=%d,y=%d)\n", mx, nd2, wb, wx, wy);
    }
    // 3. fft_fwd
    run_fwd(e_in, e_spec);
    // 4. gridfraction + apply2d
    api.SetKernelArg(kgf, 0, sizeof(cl_mem), &e_spec);
    api.SetKernelArg(kgf, 4, sizeof(cl_mem), &e_gf);
    api.EnqueueNDRangeKernel(ctx.queue, kgf, 1, nullptr, g1, nullptr, 0, nullptr, nullptr);
    api.SetKernelArg(kap, 0, sizeof(cl_mem), &e_spec);
    api.SetKernelArg(kap, 1, sizeof(cl_mem), &e_gf);
    api.EnqueueNDRangeKernel(ctx.queue, kap, 1, nullptr, gapp, nullptr, 0, nullptr, nullptr);
    // 5. fft_inv
    run_inv(e_spec, e_rinv);
    // 6. overlap_to_cover
    api.SetKernelArg(koc, 0, sizeof(cl_mem), &e_rinv);
    api.SetKernelArg(koc, 1, sizeof(cl_mem), &e_cover2);
    api.EnqueueNDRangeKernel(ctx.queue, koc, 2, nullptr, g2, nullptr, 0, nullptr, nullptr);
    // 7. cover_to_frame
    cl_kernel kcf = ctx.kernel("cover_to_frame", &e);
    if (e != CL_SUCCESS) { fprintf(stderr, "kernel cover_to_frame: %s\n", ocl::err_str(e).c_str()); return 1; }
    int dst_pitch = pw * bpp;
    api.SetKernelArg(kcf, 0, sizeof(cl_mem), &e_cover2);
    api.SetKernelArg(kcf, 1, sizeof(int), &coverpitch);
    api.SetKernelArg(kcf, 2, sizeof(cl_mem), &e_dst);
    api.SetKernelArg(kcf, 3, sizeof(int), &dst_pitch);
    api.SetKernelArg(kcf, 4, sizeof(int), &pw);
    api.SetKernelArg(kcf, 5, sizeof(int), &ph);
    api.SetKernelArg(kcf, 6, sizeof(int), &mirw);
    api.SetKernelArg(kcf, 7, sizeof(int), &mirh);
    api.SetKernelArg(kcf, 8, sizeof(int), &interlaced);
    size_t g3[2] = { (size_t)pw, (size_t)ph };
    api.EnqueueNDRangeKernel(ctx.queue, kcf, 2, nullptr, g3, nullptr, 0, nullptr, nullptr);
    api.Finish(ctx.queue);

    std::vector<unsigned char> gotdst((size_t)pw * ph * bpp);
    api.EnqueueReadBuffer(ctx.queue, e_dst, CL_TRUE, 0, gotdst.size(), gotdst.data(), 0, nullptr, nullptr);
    long nd = 0; int mx = 0; double sum2 = 0;
    if (bpp == 4) {
      // float: report ULP distance (expect <= a few ULP from FFT rounding)
      float mxa = 0.0f; int mxulp = 0;
      for (int y = 0; y < ph; y++)
        for (int x = 0; x < pw; x++) {
          size_t i = (size_t)y * pw * bpp + (size_t)x * bpp;
          size_t j = ((size_t)(ct + y) * planeW + (cl + x)) * bpp;
          float gf, rf;
          memcpy(&gf, &gotdst[i], 4); memcpy(&rf, &outr[j], 4);
          if (gf == rf) continue;
          nd++;
          float d = std::fabs(gf - rf);
          if (d > mxa) mxa = d;
          int gi, ri; memcpy(&gi, &gf, 4); memcpy(&ri, &rf, 4);
          int ulp = std::abs(gi - ri); // monotonic for same-sign floats
          if (ulp > mxulp) mxulp = ulp;
        }
      printf("END-TO-END vs out.raw: ndiff=%ld (%.2f%%) maxabs=%.3e maxulp=%d\n",
             nd, 100.0 * nd / (pw * ph), (double)mxa, mxulp);
    } else {
      for (int y = 0; y < ph; y++)
        for (int x = 0; x < pw; x++) {
          size_t i = (size_t)y * pw * bpp + (size_t)x * bpp;
          size_t j = ((size_t)(ct + y) * planeW + (cl + x)) * bpp;
          long g = 0, r = 0;
          for (int b = 0; b < bpp; b++) { g |= (long)gotdst[i + b] << (8 * b); r |= (long)outr[j + b] << (8 * b); }
          long d = std::abs(g - r);
          if (d > 0) {
            if (nd < 8) printf("  e2e diff at (x=%d,y=%d) got=%ld ref=%ld\n", x, y, g, r);
            nd++;
          }
          if (d > mx) mx = (int)d;
          sum2 += (double)d * d;
        }
      printf("END-TO-END vs out.raw: ndiff=%ld (%.2f%%) maxabs=%d rms=%.4f\n",
             nd, 100.0 * nd / (pw * ph), mx, std::sqrt(sum2 / (pw * ph)));
    }

    api.ReleaseMemObject(e_frame); api.ReleaseMemObject(e_cover); api.ReleaseMemObject(e_in);
    api.ReleaseMemObject(e_spec); api.ReleaseMemObject(e_gf); api.ReleaseMemObject(e_rinv);
    api.ReleaseMemObject(e_cover2); api.ReleaseMemObject(e_dst);
    api.ReleaseKernel(kcf);
  }

  api.ReleaseMemObject(d_in); api.ReleaseMemObject(d_out); api.ReleaseMemObject(d_fft); api.ReleaseMemObject(d_fft2);
  api.ReleaseMemObject(d_rinv); api.ReleaseMemObject(d_twR); api.ReleaseMemObject(d_twC); api.ReleaseMemObject(d_tmp);
  if (k) api.ReleaseKernel(k);
  if (ki) api.ReleaseKernel(ki);
  if (kfr) { api.ReleaseKernel(kfr); api.ReleaseKernel(kfc2); api.ReleaseKernel(kir); api.ReleaseKernel(kic); }
  return 0;
}
