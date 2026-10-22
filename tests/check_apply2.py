#!/usr/bin/env python3
"""Infer the effective Wiener factor the CPU used: f = (app-gc)/(fft-gc)."""
import numpy as np

BW = BH = 16; OUTPITCH = 16; OUTWIDTH = 9; NB = 64
fft = np.fromfile("tests/dump_bt1/p0_f000_fft.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
app = np.fromfile("tests/dump_bt1/p0_f000_applied.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
gs = np.fromfile("tests/dump_bt1/p0_f-01_gridsample.bin", dtype="<c8")[:BH * OUTPITCH].reshape(BH, OUTPITCH)[..., :OUTWIDTH]

gf = fft[:, 0, 0].real / gs[0, 0].real   # degrid=1
gc = gf[:, None, None, None] * gs[None]
z = fft - gc
f = (app - gc) / z
# per-bin implied noise: psd*(1-f)/f
psd = (z.real**2 + z.imag**2) + 1e-15
noise_implied = psd * (1 - f) / f

# sample interior bins (not DC row)
sel = f[:, 1:, 1:3]
ns = noise_implied[:, 1:, 1:3]
print("factor range:", f.min(), f.max())
print("implied noise percentiles:", np.percentile(ns.real, [1, 25, 50, 75, 99]))
# check f is real
print("imag(f) max:", np.abs(f.imag).max())
# compare against candidate constants
for c in [1024.0, 1024.0/256, 1024.0*256, 4.0, 0.015625]:
    pred = np.maximum((psd - c) / psd, 0.0)
    d = np.abs(pred - f.real)
    print("c=%10.5f  max|f-pred|=%.4f" % (c, d.max()))
