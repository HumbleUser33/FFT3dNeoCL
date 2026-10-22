import vapoursynth as vs
core = vs.core
core.std.LoadPlugin(r"C:\Users\Bruno\Desktop\OpenCode\FFT3DNEOCL\reference\build\Release\neo-fft3d.dll")
clip = core.std.BlankClip(width=320, height=240, format=vs.YV12, length=10)
clip = core.neo_fft3d.FFT3D(clip, sigma=2.0, bt=3)
clip.set_output()
