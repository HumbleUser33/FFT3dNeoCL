# Grille 2x2 etiquetee a partir de fichiers bruts arbitraires (YUV420P10 4K).
# Reutilise les fonctions ffmpeg de bench_export_png. Utile pour valider
# visuellement les reglages apparies (sigma/sharpen) du balayage match.
# Usage: python tools/crop_grid_raws.py <image> <x> <y> <taille> <out.png> ^
#          "label1=chemin1.raw" "label2=chemin2.raw" ...
# <image> = index d'image dans chaque fichier brut.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_export_png as be  # noqa: E402

TMP = os.path.join(os.environ.get("TEMP", "."), "opencode", "grid_raws")
FRAME_BYTES = 3840 * 2160 * 3  # YUV420P10 : 2 octets par echantillon


def main() -> int:
    img = int(sys.argv[1])
    x, y, size = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = sys.argv[5]
    items = [a.split("=", 1) for a in sys.argv[6:]]
    os.makedirs(TMP, exist_ok=True)
    be.CROP = size  # crop_frame lit la taille globale au moment de l'appel
    pngs, labels = [], []
    for i, (lab, path) in enumerate(items):
        with open(path, "rb") as f:
            f.seek(img * FRAME_BYTES)
            fb = f.read(FRAME_BYTES)
        raw = os.path.join(TMP, f"p{i}.raw")
        with open(raw, "wb") as f:
            f.write(be.crop_frame(fb, x, y))
        png = os.path.join(TMP, f"p{i}.png")
        be.ffmpeg_raw_png(raw, png, size, size)
        pngs.append(png)
        labels.append(lab)
    be.ffmpeg_grid(pngs, labels, out, f"{size}x{size}", 24)
    print(f"OK {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
