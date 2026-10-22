#!/usr/bin/env python3
"""Check FFTW c2r semantics: reconstruct full spectrum from half-spectrum dump,
naive unnormalized 2D inverse, compare with CPU 'inv' dump."""
import numpy as np

BW = BH = 16
OUTPITCH = 16
NB = 64

fft = np.fromfile("tests/dump_bt1/p0_f000_fft.bin", dtype="<c8")
inv = np.fromfile("tests/dump_bt1/p0_f000_inv.bin", dtype="<f4")
inp = np.fromfile("tests/dump_bt1/p0_f000_in.bin", dtype="<f4")

# layout: block, h (BH rows), w (OUTPITCH complex slots, only 0..BW/2 used)
Z = fft.reshape(NB, BH, OUTPITCH)[..., :BW // 2 + 1]

# naive unnormalized inverse of the implied full spectrum
def ifft2_full(Zhalf):
    # build full spectrum with Hermitian symmetry
    F = np.zeros((BH, BW), dtype=np.complex128)
    F[:, :BW // 2 + 1] = Zhalf
    for h in range(BH):
        for w in range(BW // 2 + 1, BW):
            F[h, w] = np.conj(F[(BH - h) % BH, BW - w])
    # unnormalized inverse: sum Z e^{+2pi i ...}
    return np.fft.ifft2(F) * (BW * BH)

ref = inv.reshape(NB, BH, BW)

b = 0
got = ifft2_full(Z[b].astype(np.complex128))
d = np.abs(got - ref[b])
print("block0 hermitian-both-idx: maxabs=%.3e" % d.max())

# alternative: symmetry only in w (what I originally coded)
def ifft2_wonly(Zhalf):
    F = np.zeros((BH, BW), dtype=np.complex128)
    F[:, :BW // 2 + 1] = Zhalf
    for h in range(BH):
        for w in range(BW // 2 + 1, BW):
            F[h, w] = np.conj(F[h, BW - w])
    return np.fft.ifft2(F) * (BW * BH)

got2 = ifft2_wonly(Z[b].astype(np.complex128))
print("block0 hermitian-w-only:   maxabs=%.3e" % np.abs(got2 - ref[b]).max())

# also check: does FFTW's c2r ignore imag at w=0/BW/2 for ALL h?
Zc = Z[b].astype(np.complex128).copy()
Zc[:, 0] = Zc[:, 0].real + 0j
Zc[:, BW // 2] = Zc[:, BW // 2].real + 0j
got3 = ifft2_full(Zc)
print("block0 imag-zero-all-rows: maxabs=%.3e" % np.abs(got3 - ref[b]).max())

# sanity: forward of input vs dump
f = np.fft.rfft2(inp.reshape(NB, BH, BW)[b]) * 1.0
print("fwd sanity block0: maxabs=%.3e" % np.abs(f - Z[b]).max())
