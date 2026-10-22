# Grille 2x2 (Original / neo-fft3d CPU / fft3dneocl GPU / fft3dcl) d'un
# recadrage a position FIXE, pour une image absolue donnee — jugement visuel
# cible (ex. le cadran avec les chiffres 500/600 de l'image 174).
# Usage: python tools/bench_crop_compare.py <seq> <image_absolue> <x0> <y0> <taille> [P1,P2,...]
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_export_png as be


def main() -> int:
    seq, fabs = int(sys.argv[1]), int(sys.argv[2])
    x0, y0, size = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    presets = sys.argv[6].split(",") if len(sys.argv) > 6 else list(be.PRESETS)
    be.CROP = size  # crop_frame et ffmpeg_grid lisent cette constante
    frame = fabs - seq
    if frame < 0 or frame >= be.FR:
        print(f"image {fabs} hors de la sequence {seq} ({seq}..{seq + be.FR - 1})")
        return 2

    for pname in presets:
        f_cpu, f_neo, f_cl = be.PRESETS[pname]
        paths = {"orig": os.path.join(be.DIR, f"seq_{seq}.raw"),
                 "cpu": os.path.join(be.DIR, f"seq{seq}", f_cpu),
                 "neocl": os.path.join(be.DIR, f"seq{seq}", f_neo)}
        if f_cl:
            paths["cl"] = os.path.join(be.DIR, f"seq{seq}", f_cl)
        tag = f"seq{seq}_{pname}_f{fabs}_zone"

        def grab(key):
            with open(paths[key], "rb") as f:
                f.seek(frame * be.FRAME_BYTES)
                return f.read(be.FRAME_BYTES)

        vers = ["orig", "cpu", "neocl"] + (["cl"] if f_cl else [])
        crops = []
        for k in vers:
            raw = os.path.join(be.TMP, f"{tag}_{k}.raw")
            with open(raw, "wb") as f:
                f.write(be.crop_frame(grab(k), x0, y0))
            png = os.path.join(be.OUT, f"{tag}_{k}.png")
            be.ffmpeg_raw_png(raw, png, size, size)
            crops.append(png)
        labs = list(be.LABELS)
        if not f_cl:
            labs[3] = "fft3dcl absent (bt=5 non supporte)"
            cph = os.path.join(be.TMP, f"{tag}_ph.raw")
            with open(cph, "wb") as f:
                f.write(b"".join([np.full(size * size, 512, dtype=np.uint16).astype("<u2").tobytes(),
                                  np.full((size // 2) ** 2, 512, dtype=np.uint16).astype("<u2").tobytes(),
                                  np.full((size // 2) ** 2, 512, dtype=np.uint16).astype("<u2").tobytes()]))
            pcph = os.path.join(be.OUT, f"{tag}_ph.png")
            be.ffmpeg_raw_png(cph, pcph, size, size)
            crops.append(pcph)
        be.ffmpeg_grid(crops, labs, os.path.join(be.OUT, f"{tag}.png"), "0x0", 24)
        print(f"OK {tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
