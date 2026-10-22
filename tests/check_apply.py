#!/usr/bin/env python3
"""Reproduce apply2d in numpy from dumps; compare with CPU 'applied' dump."""
import numpy as np

BW = BH = 16; OUTPITCH = 16; OUTWIDTH = 9; NB = 64
fft = np.fromfile("tests/dump_bt1/p0_f000_fft.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
app = np.fromfile("tests/dump_bt1/p0_f000_applied.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]
gs = np.fromfile("tests/dump_bt1/p0_f-01_gridsample.bin", dtype="<c8")[:BH * OUTPITCH].reshape(BH, OUTPITCH)[..., :OUTWIDTH]
norm = 0.00390625
sigma2N = 1024.0
gf = 1.0 * fft[:, 0, 0].real / gs[0, 0].real   # per block
print("gridfraction per block (first 4):", gf[:4])

cr = fft.real - gf[:, None, None] * gs.real[None]
ci = fft.imag - gf[:, None, None] * gs.imag[None]
psd = cr * cr + ci * ci + 1e-15
factor = np.maximum((psd - sigma2N) / psd, 0.0)
got = (cr + 1j * ci) * factor + gf[:, None, None] * gs[None]

d = np.abs(got - app)
print("formula vs applied: maxabs=%.3e rms=%.3e" % (d.max(), np.sqrt((d * d).mean())))
# where largest?
idx = np.unravel_index(np.argmax(d), d.shape)
print("worst at", idx, "got", got[idx], "ref", app[idx])
