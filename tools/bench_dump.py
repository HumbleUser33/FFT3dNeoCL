# Dump two 60-frame sequences of Sample.mkv to raw files (packed planes, no
# padding), the layout expected by fft3d_ref --in and plugin_smoke.
# Usage: python tools/bench_dump.py [outdir]
import ctypes
import os
import sys

import vapoursynth as vs

SRC = r"D:\Test Monkeys\Sample.mkv"
STARTS = [171, 419]
NFRAMES = 60


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "bench_real")
    os.makedirs(outdir, exist_ok=True)

    core = vs.core
    src = core.bs.VideoSource(SRC)
    fmt = src.format
    print(f"source: {fmt.name} {src.width}x{src.height} {src.num_frames} frames")
    if fmt.name != "YUV420P10":
        print("ERROR: expected YUV420P10")
        return 1

    w, h = src.width, src.height
    sw, sh = fmt.subsampling_w, fmt.subsampling_h
    for start in STARTS:
        clip = core.std.Trim(src, start, length=NFRAMES)
        out = os.path.join(outdir, f"seq_{start}.raw")
        with open(out, "wb") as fo:
            for i in range(NFRAMES):
                f = clip.get_frame(i)
                for p in range(3):
                    pw = (w if p == 0 else w >> sw) * fmt.bytes_per_sample
                    ph = h if p == 0 else h >> sh
                    stride = f.get_stride(p)
                    ptr = f.get_read_ptr(p)
                    if stride == pw:
                        fo.write(ctypes.string_at(ptr, pw * ph))
                    else:
                        base = ctypes.string_at(ptr, stride * ph)
                        for y in range(ph):
                            fo.write(base[y * stride:y * stride + pw])
        size = os.path.getsize(out)
        print(f"OK {out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
