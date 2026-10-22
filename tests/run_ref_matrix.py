#!/usr/bin/env python3
"""Generate CPU reference outputs for neo_FFT3D via the standalone harness.

Runs fft3d_ref.exe over a parameter matrix, checks run-to-run determinism,
and reports input/output difference statistics. Outputs land in tests/ref/.
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(ROOT, "..", "reference", "harness", "build", "Release", "fft3d_ref.exe")
OUTDIR = os.path.join(ROOT, "ref")

W, H, F, SEED = 160, 96, 7, 42

# name -> extra args
CONFIGS = {
    "default":            [],
    "bt_minus1_sharp":    ["--bt", "-1", "--sharpen", "0.5"],
    "bt0_kalman":         ["--bt", "0"],
    "bt1":                ["--bt", "1"],
    "bt2":                ["--bt", "2"],
    "bt4":                ["--bt", "4"],
    "bt5":                ["--bt", "5"],
    "wintype1":           ["--wintype", "1"],
    "wintype2":           ["--wintype", "2"],
    "sigmas":             ["--sigma", "3", "--sigma2", "2", "--sigma3", "1.5", "--sigma4", "1"],
    "pattern":            ["--pfactor", "1", "--pframe", "2"],
    "pshow":              ["--pfactor", "1", "--pframe", "2", "--pshow", "1"],
    "degrid0":            ["--degrid", "0"],
    "sharp_dehalo":       ["--sharpen", "0.5", "--dehalo", "1.0"],
    "crop":               ["--l", "10", "--t", "5", "--r", "8", "--b", "6"],
    "interlaced":         ["--interlaced", "1"],
    "bw16":               ["--bw", "16", "--bh", "16", "--ow", "5", "--oh", "5"],
    "bw64":               ["--bw", "64", "--bh", "64"],
    "beta2":              ["--beta", "2"],
    "kratio1":            ["--bt", "0", "--kratio", "1.0"],
    "opt1_c":             ["--opt", "1"],
    "opt2_sse":           ["--opt", "2"],
    "opt3_avx":           ["--opt", "3"],
    "bit10":              ["--bitdepth", "10"],
    "bit16":              ["--bitdepth", "16"],
    "bit32f":             ["--bitdepth", "32"],
    "gray":               ["--format", "GRAY"],
    "yuv444":             ["--format", "YUV444"],
    "rgbp":               ["--format", "RGBP"],
    "planes_y_only":      ["--planes", "0"],
    "sigma0":             ["--sigma", "0"],
}

BASE = ["--width", str(W), "--height", str(H), "--frames", str(F),
        "--format", "YUV420", "--bitdepth", "8", "--seed", str(SEED)]


def run(args, out):
    cmd = [EXE, "--out", out] + BASE + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"harness failed: {' '.join(args)}")
    return r.stdout.strip()


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stats(path):
    data = open(path, "rb").read()
    n = len(data)
    s = sum(data)
    return n, s / max(n, 1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # dump the input frames once (same generator, same seed)
    run(["--dump-input"], os.path.join(OUTDIR, "input.raw"))

    print(f"{'config':<18} {'bytes':>8} {'mean':>7}  determinism")
    for name, args in CONFIGS.items():
        fmt_args = [a for a in BASE]
        # per-config format overrides
        out = os.path.join(OUTDIR, name + ".raw")
        out2 = os.path.join(OUTDIR, name + ".rerun.raw")
        run(args, out)
        run(args, out2)
        det = "OK" if sha(out) == sha(out2) else "MISMATCH"
        os.remove(out2)
        n, mean = stats(out)
        print(f"{name:<18} {n:>8} {mean:>7.1f}  {det}")

    print("\nAll reference outputs in", OUTDIR)


if __name__ == "__main__":
    main()
