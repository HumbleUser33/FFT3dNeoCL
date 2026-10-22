# Quality comparison between two raw outputs (packed planes, no padding).
# Prints ndiff, maxabs, MAE and PSNR per plane and overall, as JSON on stdout.
# Usage: python tools/bench_compare.py A.raw B.raw W H bits frames [ssw ssh]
import json
import sys

import numpy as np


def main() -> int:
    a_path, b_path = sys.argv[1], sys.argv[2]
    w, h, bits, frames = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    ssw = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    ssh = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    bpp = 1 if bits <= 8 else (2 if bits <= 16 else 4)
    dt = np.uint8 if bpp == 1 else (np.uint16 if bpp == 2 else np.float32)

    planes = [(w, h)]
    if ssw or ssh:
        planes += [(w >> ssw, h >> ssh), (w >> ssw, h >> ssh)]
    sizes = [pw * ph for pw, ph in planes]
    frame_bytes = sum(sizes) * bpp

    fa, fb = open(a_path, "rb"), open(b_path, "rb")
    peak = float((1 << bits) - 1)
    tot_n = tot_diff = 0
    tot_max = 0
    tot_sse = tot_ae = 0.0
    per_plane = []
    for pi, (pw, ph) in enumerate(planes):
        n = sizes[pi] * frames
        a = np.frombuffer(fa.read(n * bpp), dtype=dt).astype(np.float64)
        b = np.frombuffer(fb.read(n * bpp), dtype=dt).astype(np.float64)
        if a.size != n or b.size != n:
            print(f"ERROR: short file on plane {pi}")
            return 1
        d = np.abs(a - b)
        nd = int(np.count_nonzero(d))
        mx = float(d.max()) if n else 0.0
        mae = float(d.mean()) if n else 0.0
        mse = float((d * d).mean()) if n else 0.0
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)
        per_plane.append({"plane": pi, "ndiff": nd, "npix": n, "pct": 100.0 * nd / n,
                          "maxabs": mx, "mae": mae, "psnr": psnr})
        tot_n += n
        tot_diff += nd
        tot_max = max(tot_max, mx)
        tot_sse += mse * n
        tot_ae += mae * n
    fa.close()
    fb.close()
    tmse = tot_sse / tot_n
    result = {
        "a": a_path, "b": b_path,
        "ndiff": tot_diff, "npix": tot_n, "pct": 100.0 * tot_diff / tot_n,
        "maxabs": tot_max, "mae": tot_ae / tot_n,
        "psnr": None if tmse == 0 else round(10.0 * np.log10(peak * peak / tmse), 2),
        "identical": tot_diff == 0,
        "planes": per_plane,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
