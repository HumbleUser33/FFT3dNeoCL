# Export d'images PNG pour jugement visuel du comparatif (bench_real).
# Pour chaque sequence et chaque preset : frame 30 de la sequence (image 201 /
# 449 absolue), en 4 versions (original, neo-fft3d CPU, fft3dneocl GPU,
# fft3dcl) : images completes 4K + grille 2x2 etiquetee + grille 2x2 de
# recadrages 512x512 takenus dans la zone de plus fort desaccord neocl/cl.
# Usage: python tools/bench_export_png.py [frame]
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "tests", "bench_real")
OUT = os.path.join(DIR, "png")
TMP = os.path.join(os.environ.get("TEMP", "."), "opencode", "bench_png")
W, H, FR = 3840, 2160, 60
FRAME_BYTES = W * H * 2 + 2 * (W // 2) * (H // 2) * 2
CROP = 512
FFMPEG = "ffmpeg"
FONT = "C\\\\:/Windows/Fonts/arial.ttf"  # double echappement : chemin contient un ':'

# preset -> (fichier cpu, fichier neocl, fichier cl ou None)
PRESETS = {
    "P1": ("P1_cpu.raw", "P1_neocl.raw", "P1_cl.raw"),
    "P2": ("P2_cpu.raw", "P2_neocl.raw", "P2_cl.raw"),
    "P3": ("P3_cpu.raw", "P3_neocl.raw", "P3_cl.raw"),
    "P4": ("P4_cpu.raw", "P4_neocl.raw", None),
    "P5": ("P5_cpu_p2.raw", "P5_neocl_p2.raw", "P5_cl_p2.raw"),
}
LABELS = ["Original", "neo-fft3d CPU", "fft3dneocl GPU", "fft3dcl"]


def luma(frame_bytes: bytes) -> np.ndarray:
    return np.frombuffer(frame_bytes[:W * H * 2], dtype=np.uint16).reshape(H, W)


def worst_crop(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """position x0,y0 (pairs) du bloc CROP x CROP de plus fort desaccord moyen"""
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    # moyennes par bloc via sommes integralles, recherche pas 128
    cs = np.zeros((H + 1, W + 1), dtype=np.int64)
    cs[1:, 1:] = d.cumsum(axis=0).cumsum(axis=1)
    best, bx, by = -1.0, 0, 0
    for y in range(0, H - CROP + 1, 128):
        for x in range(0, W - CROP + 1, 128):
            s = cs[y + CROP, x + CROP] - cs[y, x + CROP] - cs[y + CROP, x] + cs[y, x]
            m = s / (CROP * CROP)
            if m > best:
                best, bx, by = m, x, y
    return bx & ~1, by & ~1


def crop_frame(fb: bytes, x0: int, y0: int) -> bytes:
    y = np.frombuffer(fb[:W * H * 2], dtype=np.uint16).reshape(H, W)
    y = y[y0:y0 + CROP, x0:x0 + CROP]
    off = W * H * 2
    cw = W // 2
    planes = [y]
    for p in range(2):
        base = off + p * cw * (H // 2) * 2
        u = np.frombuffer(fb[base:base + cw * (H // 2) * 2], dtype=np.uint16).reshape(H // 2, cw)
        planes.append(u[y0 // 2:y0 // 2 + CROP // 2, x0 // 2:x0 // 2 + CROP // 2])
    return b"".join(p.astype("<u2").tobytes() for p in planes)


def ffmpeg_raw_png(raw: str, png: str, w: int, h: int) -> None:
    subprocess.run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-video_size", f"{w}x{h}",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-i", raw,
        "-vf", "scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709,format=rgb24",
        "-frames:v", "1", png,
    ], check=True)


def ffmpeg_grid(pngs: list[str], labels: list[str], out: str, panel: str, fs: int) -> None:
    """grille 2x2 : haut gauche/droite, bas gauche/droite ; etiquette sur chaque panneau"""
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in pngs:
        args += ["-i", p]
    if panel != "0x0":
        pw, ph = (int(v) for v in panel.split("x"))
        scale = f"scale={pw}:{ph}:flags=area"
    else:
        pw, ph = CROP, CROP
        scale = ""
    if scale:
        fc = (
            f"[0:v]{scale}[a];[1:v]{scale}[b];[a][b]hstack[t];"
            f"[2:v]{scale}[c];[3:v]{scale}[d];[c][d]hstack[e];[t][e]vstack[g]"
        )
    else:
        fc = "[0:v][1:v]hstack[t];[2:v][3:v]hstack[e];[t][e]vstack[g]"
    pos = [(30, 30), (pw + 30, 30), (30, ph + 30), (pw + 30, ph + 30)]
    prev = "g"
    for i, lab in enumerate(labels):
        x, y = pos[i]
        nxt = "out" if i == len(labels) - 1 else f"d{i}"
        fc += (f";[{prev}]drawtext=fontfile={FONT}:text='{lab}':x={x}:y={y}"
               f":fontsize={fs}:fontcolor=white:box=1:boxcolor=black@0.6[{nxt}]")
        prev = nxt
    args += ["-filter_complex", fc, "-map", "[out]", "-frames:v", "1", out]
    subprocess.run(args, check=True)


def main() -> int:
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    for seq in (171, 419):
        n_abs = seq + frame
        for pname, (f_cpu, f_neo, f_cl) in PRESETS.items():
            tag = f"seq{seq}_{pname}_f{n_abs}"
            paths = {"orig": os.path.join(DIR, f"seq_{seq}.raw"),
                     "cpu": os.path.join(DIR, f"seq{seq}", f_cpu),
                     "neocl": os.path.join(DIR, f"seq{seq}", f_neo)}
            if f_cl:
                paths["cl"] = os.path.join(DIR, f"seq{seq}", f_cl)

            def grab(key):
                with open(paths[key], "rb") as f:
                    f.seek(frame * FRAME_BYTES)
                    return f.read(FRAME_BYTES)

            fbs = {k: grab(k) for k in paths}
            # zone de plus fort desaccord (neocl vs cl, sinon neocl vs original)
            ref_b = fbs["cl"] if f_cl else fbs["orig"]
            x0, y0 = worst_crop(luma(fbs["neocl"]), luma(ref_b))
            vers = ["orig", "cpu", "neocl"] + (["cl"] if f_cl else [])
            fulls = []
            for k in vers:
                raw = os.path.join(TMP, f"{tag}_{k}.raw")
                with open(raw, "wb") as f:
                    f.write(fbs[k])
                png = os.path.join(OUT, f"{tag}_{k}.png")
                ffmpeg_raw_png(raw, png, W, H)
                fulls.append(png)
            # recadrages 512x512
            crops = []
            for k in vers:
                raw = os.path.join(TMP, f"{tag}_{k}_crop.raw")
                with open(raw, "wb") as f:
                    f.write(crop_frame(fbs[k], x0, y0))
                png = os.path.join(OUT, f"{tag}_{k}_crop.png")
                ffmpeg_raw_png(raw, png, CROP, CROP)
                crops.append(png)
            # grilles 2x2 (cas echant : panneau gris si fft3dcl absent)
            labs = list(LABELS)
            grid_in = fulls[:]
            crop_in = crops[:]
            if not f_cl:
                labs[3] = "fft3dcl absent (bt=5 non supporte)"
                ph = os.path.join(TMP, f"{tag}_ph.raw")
                with open(ph, "wb") as f:  # gris YUV 10 bits
                    f.write(b"".join([(np.full(W * H, 512, dtype=np.uint16)).astype("<u2").tobytes(),
                                      (np.full((W // 2) * (H // 2), 512, dtype=np.uint16)).astype("<u2").tobytes(),
                                      (np.full((W // 2) * (H // 2), 512, dtype=np.uint16)).astype("<u2").tobytes()]))
                pph = os.path.join(TMP, f"{tag}_ph.png")
                ffmpeg_raw_png(ph, pph, W, H)
                grid_in.append(pph)
                cph = os.path.join(TMP, f"{tag}_ph_crop.raw")
                with open(cph, "wb") as f:
                    f.write(b"".join([(np.full(CROP * CROP, 512, dtype=np.uint16)).astype("<u2").tobytes(),
                                      (np.full((CROP // 2) ** 2, 512, dtype=np.uint16)).astype("<u2").tobytes(),
                                      (np.full((CROP // 2) ** 2, 512, dtype=np.uint16)).astype("<u2").tobytes()]))
                pcph = os.path.join(TMP, f"{tag}_ph_crop.png")
                ffmpeg_raw_png(cph, pcph, CROP, CROP)
                crop_in.append(pcph)
            ffmpeg_grid(grid_in, labs, os.path.join(OUT, f"{tag}_grille.png"), "1920x1080", 56)
            ffmpeg_grid(crop_in, labs, os.path.join(OUT, f"{tag}_recadrage.png"), "0x0", 24)
            print(f"OK {tag}  recadrage en x={x0} y={y0}")
    print(f"\nImages ecrites dans {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
