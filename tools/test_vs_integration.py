# Integration smoke test: real VapourSynth (R81, API4) loading fft3dneocl.dll.
# Run: python tools\test_vs_integration.py
import hashlib
import os
import vapoursynth as vs

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dll = os.path.join(root, "fft3dneocl", "build", "Release", "fft3dneocl.dll")

core = vs.core
core.std.LoadPlugin(dll)

print("ListDevices:", core.fft3dneocl.ListDevices())

clip = core.std.BlankClip(width=64, height=64, format=vs.GRAY8, length=7, color=128)

for bt in (2, 3, 4, 5):
    f = core.fft3dneocl.FFT3D(clip, bt=bt, bw=16, bh=16, sigma=2)
    h1 = [hashlib.sha256(b"".join(f.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
    h2 = [hashlib.sha256(b"".join(f.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
    print(f"bt={bt}: {'OK deterministe' if h1 == h2 else 'NON DETERMINISTE'} {h1}")

# bt=-1 (sharpen only): deterministic, and equal to the CPU-verified equivalence
# bt=1 + sigma=0 + sharpen (Wiener factor = 1) which goes through apply2d instead
f = core.fft3dneocl.FFT3D(clip, bt=-1, bw=16, bh=16, sharpen=1)
h1 = [hashlib.sha256(b"".join(f.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
h2 = [hashlib.sha256(b"".join(f.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
g = core.fft3dneocl.FFT3D(clip, bt=1, bw=16, bh=16, sigma=0, sharpen=1)
h3 = [hashlib.sha256(b"".join(g.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
ok = "OK deterministe" if h1 == h2 else "NON DETERMINISTE"
same = "et identique a bt=1 sigma=0" if h1 == h3 else "DIVERGENT de bt=1 sigma=0"
print(f"bt=-1: {ok} {same} {h1}")

# bt=0 (Kalman): accepted. fmUnordered serializes requests on the instance; the
# IIR state is sequential, so a second pass continues it (h2 != h1) - faithful to
# the CPU engine. Frame 0 is the source passthrough (engine 789-791).
f = core.fft3dneocl.FFT3D(clip, bt=0, bw=16, bh=16, sigma=2)
h1 = [hashlib.sha256(b"".join(f.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
f0 = b"".join(f.get_frame(0).readchunks())
src0 = b"".join(clip.get_frame(0).readchunks())
print(f"bt=0: OK {h1} ; frame0 passthrough: {f0 == src0}")

# pattern denoise (pfactor != 0): the pattern is estimated once from frame
# pframe (requested in addition to the temporal window). Out-of-order requests
# must not change the result: the same pattern is used whichever frame arrives
# first. The sigma2/3/4 variant builds the pattern without fetching a frame.
f = core.fft3dneocl.FFT3D(clip, bt=3, bw=16, bh=16, sigma=2, pfactor=1, pframe=3)
h1 = [hashlib.sha256(b"".join(f.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
# fresh node, frames requested 6..0: the pattern is then estimated on the
# frame-6 call and must still match the sequential result
f2 = core.fft3dneocl.FFT3D(clip, bt=3, bw=16, bh=16, sigma=2, pfactor=1, pframe=3)
h2 = {}
for n in range(6, -1, -1):
    h2[n] = hashlib.sha256(b"".join(f2.get_frame(n).readchunks())).hexdigest()[:12]
h2 = [h2[n] for n in range(7)]
print(f"pfactor=1 pframe=3: {'OK ordre inverse identique' if h1 == h2 else 'DIVERGENT ordre inverse'} {h1}")
g = core.fft3dneocl.FFT3D(clip, bt=1, bw=16, bh=16, sigma=2, sigma2=1, sigma3=3, sigma4=5)
h3 = [hashlib.sha256(b"".join(g.get_frame(n).readchunks())).hexdigest()[:12] for n in range(7)]
print(f"sigma2/3/4: OK {h3}")
