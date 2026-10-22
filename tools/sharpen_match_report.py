# Analyse du balayage sharpen (tools/run_sigma_match.ps1 -Phase B).
# Pour chaque bt, aux sigma appariees trouvees en Phase A (bt=3 : 3.0/3.0,
# bt=4 : 3.0/3.5), on compare les deux filtres avec sharpen = 0.25..1.0 :
#  (1) force du sharpen = surcroit d'ecart a l'original par rapport a sharpen=0 ;
#  (2) difference directe entre sorties neocl x fft3dcl (matrice 4x4).
# Usage: python tools/sharpen_match_report.py
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

# sigma appariees par bt (resultat Phase A, critere difference minimale)
MATCH = {3: (3.0, 3.0), 4: (3.0, 3.5)}
SH = [0.25, 0.5, 0.75, 1.0]


def load(path):
    return np.memmap(path, dtype="<u2", mode="r", shape=(FR * sum(PLANE_N),))


def mae(a, b):
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float(d.sum(dtype=np.int64) / d.size)


def fmt(s):
    """nom PowerShell : 3.0 -> '3', 3.5 -> '3.5'"""
    return str(int(s)) if s == int(s) else str(s)


def interp(xs, ys, target):
    for i in range(len(xs) - 1):
        if ys[i] <= target <= ys[i + 1]:
            t = (target - ys[i]) / (ys[i + 1] - ys[i])
            return round(xs[i] + t * (xs[i + 1] - xs[i]), 2)
    return xs[0] if target < ys[0] else xs[-1]


def main():
    orig = load(SEQ)
    report = {}
    for bt, (sn, sc) in MATCH.items():
        base_n = load(os.path.join(DIR, f"bt{bt}_neo_s{fmt(sn)}.raw"))
        base_c = load(os.path.join(DIR, f"bt{bt}_cl_s{fmt(sc)}.raw"))
        m0n, m0c = mae(base_n, orig), mae(base_c, orig)
        neo, cl = {}, {}
        for a in SH:
            neo[a] = load(os.path.join(DIR, f"bt{bt}_neo_s{fmt(sn)}_k{fmt(a)}.raw"))
            cl[a] = load(os.path.join(DIR, f"bt{bt}_cl_s{fmt(sc)}_k{fmt(a)}.raw"))
        gain_n = {a: round(mae(neo[a], orig) - m0n, 3) for a in SH}
        gain_c = {a: round(mae(cl[a], orig) - m0c, 3) for a in SH}
        cross = {a: {b: round(mae(neo[a], cl[b]), 3) for b in SH} for a in SH}
        equiv = []
        for a in SH:
            best_b = min(SH, key=lambda b: cross[a][b])
            by_force = interp(SH, [gain_c[b] for b in SH], gain_n[a])
            equiv.append({"neo_sharpen": a, "cl_sharpen_min_diff": best_b,
                          "diff": cross[a][best_b], "cl_sharpen_meme_force": by_force})
        report[f"bt{bt}"] = {"sigma": {"neo": sn, "cl": sc},
                             "mae_orig_sans_sharpen": {"neo": round(m0n, 3), "cl": round(m0c, 3)},
                             "gain_sharpen": {"neo": gain_n, "cl": gain_c},
                             "cross": cross, "equivalences": equiv}
        print(f"\n=== bt={bt}  sigma neocl={sn} / fft3dcl={sc}  (sharpen) ===")
        print("sharpen      " + "".join(f"{a:>8}" for a in SH))
        print("surcroit neo " + "".join(f"{gain_n[a]:>8}" for a in SH))
        print("surcroit cl  " + "".join(f"{gain_c[a]:>8}" for a in SH))
        print("\ndifference directe neo(lignes) x cl(colonnes):")
        print("             " + "".join(f"{b:>8}" for b in SH))
        for a in SH:
            print(f"neo k={a:<5}  " + "".join(f"{cross[a][b]:>8}" for b in SH))
        print("\n equivalences:")
        for e in equiv:
            print(f"  neocl sharpen={e['neo_sharpen']:<5} -> cl sharpen={e['cl_sharpen_min_diff']:<5}"
                  f" (diff min {e['diff']})  |  cl sharpen={e['cl_sharpen_meme_force']} (meme force)")
    with open(os.path.join(DIR, "sharpen_match.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nJSON: tests\\bench_real\\match\\sharpen_match.json")


if __name__ == "__main__":
    main()
