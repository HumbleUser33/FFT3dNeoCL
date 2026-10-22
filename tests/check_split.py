#!/usr/bin/env python3
"""Simulate the split FFT (fft1d Stockham) in float32, compare with FFTW dump."""
import numpy as np

BW = BH = 16; OUTPITCH = 16; OUTWIDTH = 9; NB = 64
inp = np.fromfile("tests/dump_bt1/p0_f000_in.bin", dtype="<f4").reshape(NB, BH, BW)
fft = np.fromfile("tests/dump_bt1/p0_f000_fft.bin", dtype="<c8").reshape(NB, BH, OUTPITCH)[..., :OUTWIDTH]

def twiddle(N):
    # packed per stage: ncur=N..2, offset N-ncur, entries exp(-2pi i p/ncur), p<ncur/2
    t = []
    off = 0
    table = {}
    for ncur in range(N, 1, -1):
        if ncur & (ncur-1): continue
        for p in range(ncur//2):
            a = -2*np.pi*p/ncur
            table[(N, ncur, p)] = np.complex64(complex(np.cos(a), np.sin(a)))
    return table

def fft1d(x, N):
    x = x.astype(np.complex64).copy()
    tw = twiddle(N)
    X = x
    ncur, s = N, 1
    while ncur >= 2:
        m = ncur >> 1
        Y = np.empty_like(X)
        for idx in range(N >> 1):
            q = idx % s; p = idx // s
            a = X[q + s*p]; b = X[q + s*(p+m)]
            w = tw[(N, ncur, p)]
            Y[q + 2*s*p] = np.complex64(a + b)
            Y[q + s*(2*p+1)] = np.complex64((a - b) * w)
        X = Y
        ncur >>= 1; s <<= 1
    return X

# row pass then col pass, block 0
b = 0
tmp = np.empty((BH, BW), dtype=np.complex64)
for h in range(BH):
    tmp[h] = fft1d(inp[b, h], BW)
out = np.empty((BH, BW), dtype=np.complex64)
for w in range(BW):
    out[:, w] = fft1d(tmp[:, w], BH)

d = np.abs(out[:, :OUTWIDTH] - fft[b])
print("split-sim vs FFTW block0: maxabs=%.3e" % d.max())
# compare fused-equivalent (numpy rfft2) for sanity
f = np.fft.rfft2(inp[b])
print("numpy rfft2 vs dump: maxabs=%.3e" % np.abs(f - fft[b]).max())
