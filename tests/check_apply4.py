#!/usr/bin/env python3
import numpy as np

BW = BH = 16; OUTPITCH = 16; OUTWIDTH = 9; NB = 64
fft = np.fromfile("tests/dump_bt1/p0_f000_fft.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
app = np.fromfile("tests/dump_bt1/p0_f000_applied.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
gs = np.fromfile("tests/dump_bt1/p0_f-01_gridsample.bin", dtype="<c8")[:BH * OUTPITCH].reshape(BH, OUTPITCH)[..., :OUTWIDTH]

gf = fft[:, 0, 0].real / gs[0, 0].real
gc = gf[:, None, None] * gs[None]
z = fft - gc
psd = (z.real**2 + z.imag**2) + 1e-15

big = np.abs(z) > 200
f = np.zeros_like(fft)
f[big] = (app - gc)[big] / z[big]
print("imag(f) max over |z|>200:", np.abs(f.imag[big]).max())
print("f real range:", f.real[big].min(), f.real[big].max())

ni = psd * (1 - f.real) / np.maximum(f.real, 1e-9)
m = big & (f.real > 0.05) & (f.real < 0.95)
print("implied noise over f in (0.05,0.95): median=%.1f p5=%.1f p95=%.1f n=%d" %
      (np.median(ni[m]), np.percentile(ni[m],5), np.percentile(ni[m],95), m.sum()))

# check psd distribution: how many bins have psd < 1024 (should be zeroed -> f=0)
low = psd < 1024
print("bins with psd<1024:", low.sum(), " their f mean:", f.real[low].mean())
# and bins with f==0 exactly
print("f==0 bins:", (f.real == 0).sum(), " psd range there:", psd[f.real==0].min() if (f.real==0).any() else "-", psd[f.real==0].max() if (f.real==0).any() else "-")

# Try assuming NO degrid correction (gc=0):
z2 = fft
psd2 = (z2.real**2 + z2.imag**2) + 1e-15
f2 = np.zeros_like(fft)
b2 = np.abs(z2) > 200
f2[b2] = (app)[b2] / z2[b2]
ni2 = psd2 * (1 - f2.real) / np.maximum(f2.real, 1e-9)
m2 = b2 & (f2.real > 0.05) & (f2.real < 0.95)
print("NO-degrid implied noise: median=%.1f p5=%.1f p95=%.1f" %
      (np.median(ni2[m2]), np.percentile(ni2[m2],5), np.percentile(ni2[m2],95)))
