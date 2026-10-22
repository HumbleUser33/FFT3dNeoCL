#!/usr/bin/env python3
"""Per-bin implied Wiener noise constant, excluding degenerate bins."""
import numpy as np

BW = BH = 16; OUTPITCH = 16; OUTWIDTH = 9; NB = 64
fft = np.fromfile("tests/dump_bt1/p0_f000_fft.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
app = np.fromfile("tests/dump_bt1/p0_f000_applied.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
gs = np.fromfile("tests/dump_bt1/p0_f-01_gridsample.bin", dtype="<c8")[:BH * OUTPITCH].reshape(BH, OUTPITCH)[..., :OUTWIDTH]

gf = fft[:, 0, 0].real / gs[0, 0].real
gc = gf[:, None, None, None] * gs[None]
z = fft - gc
psd = (z.real**2 + z.imag**2) + 1e-15

# factor per bin via real projection: f = Re((app-gc)*conj(z))/|z|^2
f = ((app - gc) * np.conj(z)).real / (psd - 1e-15)
noise_implied = psd * (1 - f) / np.maximum(f, 1e-9)

mask = (f > 0.02) & (f < 0.98) & (psd > 1e4)
ni = noise_implied[mask]
print("n bins used:", mask.sum())
print("implied noise: mean=%.2f median=%.2f std=%.2f min=%.2f max=%.2f" %
      (ni.mean(), np.median(ni), ni.std(), ni.min(), ni.max()))

# mean implied noise per (h,w)
for h in [0, 1, 4, 8, 12]:
    row = []
    for w in range(OUTWIDTH):
        m = mask[:, h, w]
        row.append("%.0f" % noise_implied[:, h, w][m].mean() if m.sum() else "--")
    print("h=%2d:" % h, " ".join("%7s" % v for v in row))

# also check DC preservation
print("DC: max|app-fft| =", np.abs(app[:, 0, 0] - fft[:, 0, 0]).max())
# and f at DC
print("f at (0,0):", f[:, 0, 0][:4])
