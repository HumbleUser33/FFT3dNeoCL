# Analyse du balayage sigma (tools/run_sigma_match.ps1 -Phase A).
# Pour chaque bt : courbes de "force" (ecart a l'original) des deux filtres,
# puis sigma fft3dcl equivalente a chaque sigma fft3dneocl, par (1) egalite
# d'ecart a l'original (interpolee) et (2) minimum de difference directe.
# Usage: python tools/sigma_match_report.py
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "tests", "bench_real", "match")
SEQ = os.path.join(ROOT, "tests", "bench_real", "seq_171.raw")
W, H, FR = 3840, 2160, 10
PLANE_N = [W * H, (W // 2) * (H // 2), (W // 2) * (H // 2)]
OFFS = [0, PLANE_N[0], PLANE_N[0] + PLANE_N[1]]
PEAK = 1023.0


def load(path):
    return np.memmap(path, dtype="<u2", mode="r", shape=(FR * sum(PLANE_N),))


def stats_vs_orig(a, o):
    out = {}
    tot_sse = tot_ae = 0.0
    A = a.reshape(FR, -1)  # layout repete par image : [Y][U][V]
    O = o.reshape(FR, -1)
    for i, name in enumerate("YUV"):
        d = A[:, OFFS[i]:OFFS[i] + PLANE_N[i]].astype(np.int32) - O[:, OFFS[i]:OFFS[i] + PLANE_N[i]].astype(np.int32)
        ad = np.abs(d)
        out[name] = {"mae": float(ad.mean()), "max": int(ad.max())}
        tot_sse += float((d * d).sum(dtype=np.int64))
        tot_ae += float(ad.sum(dtype=np.int64))
    n = FR * sum(PLANE_N)
    mse = tot_sse / n
    out["all"] = {"mae": tot_ae / n,
                  "psnr": None if mse == 0 else round(10 * np.log10(PEAK * PEAK / mse), 2)}
    return out


def cross_mae(a, b):
    d = a.astype(np.int32) - b.astype(np.int32)
    return float(np.abs(d).sum(dtype=np.int64) / d.size)


def interp_sigma(xs, ys, target):
    """sigma x ou la courbe croissante ys atteint target (lineaire, bornes extrapoles)"""
    for i in range(len(xs) - 1):
        if ys[i] <= target <= ys[i + 1]:
            t = (target - ys[i]) / (ys[i + 1] - ys[i])
            return round(xs[i] + t * (xs[i + 1] - xs[i]), 2)
    if target < ys[0]:
        return xs[0]
    return xs[-1]


def main():
    orig = load(SEQ)
    report = {}
    for bt in (3, 4):
        neo = {}
        cl = {}
        for p in sorted(glob.glob(os.path.join(DIR, f"bt{bt}_neo_s*.raw"))):
            m = re.match(rf"bt{bt}_neo_s([\d.]+)\.raw", os.path.basename(p))
            if m and "_k" not in p:
                neo[float(m.group(1))] = p
        for p in sorted(glob.glob(os.path.join(DIR, f"bt{bt}_cl_s*.raw"))):
            m = re.match(rf"bt{bt}_cl_s([\d.]+)\.raw", os.path.basename(p))
            if m and "_k" not in p:
                cl[float(m.group(1))] = p
        ns, cs = sorted(neo), sorted(cl)
        curves = {"neo": [], "cl": []}
        data = {}
        for k, sig in (("neo", ns), ("cl", cs)):
            for s in sig:
                a = load(neo[s] if k == "neo" else cl[s])
                data[(k, s)] = a
                st = stats_vs_orig(a, orig)
                curves[k].append({"sigma": s, "mae_all": round(st["all"]["mae"], 3),
                                  "mae_y": round(st["Y"]["mae"], 3), "psnr_all": st["all"]["psnr"]})
        mae_neo = [c["mae_all"] for c in curves["neo"]]
        mae_cl = [c["mae_all"] for c in curves["cl"]]
        equiv = []
        for i, s in enumerate(ns):
            by_strength = interp_sigma(cs, mae_cl, mae_neo[i])
            cx = {c: cross_mae(data[("neo", s)], data[("cl", c)]) for c in cs}
            best = min(cx, key=cx.get)
            near = min(cs, key=lambda c: abs(c - by_strength))
            equiv.append({"neo_sigma": s, "cl_sigma_meme_force": by_strength,
                          "cl_sigma_min_diff": best, "diff_min": round(cx[best], 3),
                          "diff_meme_force": round(cx[near], 3)})
        report[f"bt{bt}"] = {"courbes": curves, "equivalences": equiv}
        print(f"\n=== bt={bt} : ecart a l'original (force de denoising) ===")
        print("sigma      " + "".join(f"{s:>8}" for s in ns) + "   (fft3dneocl)")
        print("MAE totale " + "".join(f"{m:>8}" for m in mae_neo))
        print("sigma      " + "".join(f"{s:>8}" for s in cs) + "   (fft3dcl)")
        print("MAE totale " + "".join(f"{m:>8}" for m in mae_cl))
        print("\n equivalences:")
        for e in equiv:
            print(f"  neocl sigma={e['neo_sigma']:<5} -> cl sigma={e['cl_sigma_meme_force']:<6} (meme force)"
                  f"  |  cl sigma={e['cl_sigma_min_diff']:<6} (diff min {e['diff_min']})")
    with open(os.path.join(DIR, "sigma_match.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nJSON: tests\\bench_real\\match\\sigma_match.json")


if __name__ == "__main__":
    main()
