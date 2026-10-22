#!/usr/bin/env python3
"""Compare two raw planar outputs (same layout) and report differences.

Usage: compare.py A.raw B.raw [--label "name"]
Reports: max abs diff, mean abs diff, % pixels differing, % pixels > 1 LSB.
"""
import sys
import numpy as np


def load(path, meta_path=None):
    meta = {}
    mp = meta_path or (path + ".meta.txt")
    import os
    if not os.path.exists(mp) and path.endswith(".rerun.raw"):
        mp = path[:-len(".rerun.raw")] + ".raw.meta.txt"
    with open(mp) as f:
        for line in f:
            k, v = line.strip().split("=")
            meta[k] = int(v)
    bps = meta["bytespersample"]
    dt = np.uint8 if bps == 1 else ("<u2" if bps == 2 else "<f4")
    data = np.fromfile(path, dtype=dt)
    return meta, data


def to_pixels(meta, data):
    return data


def plane_slices(meta):
    """Yield (start, end) sample ranges for each plane of each frame."""
    w, h = meta["width"], meta["height"]
    planes, ssw, ssh = meta["planes"], meta["ssw"], meta["ssh"]
    for p in range(planes):
        pw = w if p == 0 else (w >> ssw)
        ph = h if p == 0 else (h >> ssh)
        yield p, pw * ph


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    label = ""
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]

    ma, a = load(a_path)
    mb, b = load(b_path)
    assert ma == mb, f"meta mismatch: {ma} vs {mb}"

    pa, pb = to_pixels(ma, a), to_pixels(mb, b)
    if pa.dtype == np.float32:
        scale = 255.0  # report float diffs on a 0..1 scale times 255
    else:
        scale = 1.0

    d = np.abs(pa.astype(np.float64) - pb.astype(np.float64)) * scale
    total = d.size
    ndiff = int((d > 1e-9).sum())
    over1 = int((d > 1.0 + 1e-9).sum())
    print(f"{label or 'compare'}:")
    print(f"  samples={total}  max={d.max():.3f}  mean={d.mean():.4f}"
          f"  diff={100.0*ndiff/total:.2f}%  over1LSB={100.0*over1/total:.3f}%")


if __name__ == "__main__":
    main()
