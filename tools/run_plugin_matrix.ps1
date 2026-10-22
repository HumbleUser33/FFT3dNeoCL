# Runs the whole fft3dneocl plugin fixture matrix and prints one line per fixture.
# Usage: powershell -File tools\run_plugin_matrix.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$smoke = Join-Path $root 'fft3dneocl\build\Release\plugin_smoke.exe'
if (-not (Test-Path $smoke)) { Write-Error "missing $smoke (build Release first)" }

# name, format, bits, frames (temporal fixtures hold 7 frames), filter args,
# tol = rounding-boundary differences up to 1 LSB (float: ~1e-7) are expected
$cases = @(
  @{ n = 'dump_bt1';        f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt1_10b';    f = 'GRAY';   b = 10; a = @('bt=1', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt1_12b';    f = 'GRAY';   b = 12; a = @('bt=1', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt1_14b';    f = 'GRAY';   b = 14; tol = $true; a = @('bt=1', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt1_16b';    f = 'GRAY';   b = 16; tol = $true; a = @('bt=1', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt1_32f';    f = 'GRAY';   b = 32; tol = $true; a = @('bt=1', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt32';       f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=32', 'bh=32') },
  @{ n = 'dump_bt64';       f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=64', 'bh=64') },
  @{ n = 'dump_bt128';      f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=128', 'bh=128') },
  @{ n = 'dump_wt1';        f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'wintype=1') },
  @{ n = 'dump_wt2';        f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'wintype=2') },
  @{ n = 'dump_ilace';      f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'interlaced=1') },
  @{ n = 'dump_crop';       f = 'GRAY';   b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'l=3', 't=2', 'r=5', 'b=4') },
  @{ n = 'dump_chroma';     f = 'YUV420'; b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'planes=0,1') },
  @{ n = 'dump_chroma_f';   f = 'YUV420'; b = 32; tol = $true; a = @('bt=1', 'bw=16', 'bh=16', 'planes=0,1') },
  @{ n = 'dump_yuv422';     f = 'YUV422'; b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'planes=0,1') },
  @{ n = 'dump_yuv444';     f = 'YUV444'; b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'planes=0,1') },
  @{ n = 'dump_yuv_p0only'; f = 'YUV420'; b = 8;  a = @('bt=1', 'bw=16', 'bh=16', 'planes=0') },
  # temporal Wiener (bt=2..5), 7 frames: edge frames fall back to 2D
  @{ n = 'dump_bt2';          f = 'GRAY';   b = 8;  fr = 7; a = @('bt=2', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt3';          f = 'GRAY';   b = 8;  fr = 7; a = @('bt=3', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt4';          f = 'GRAY';   b = 8;  fr = 7; a = @('bt=4', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt5';          f = 'GRAY';   b = 8;  fr = 7; a = @('bt=5', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt3_bw32';     f = 'GRAY';   b = 8;  fr = 7; a = @('bt=3', 'bw=32', 'bh=32') },
  @{ n = 'dump_bt3_yuv';      f = 'YUV420'; b = 8;  fr = 7; a = @('bt=3', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt4_sharpen';  f = 'GRAY';   b = 8;  fr = 7; a = @('bt=4', 'bw=16', 'bh=16', 'sharpen=1', 'dehalo=1') },
  @{ n = 'dump_bt5_crop';     f = 'GRAY';   b = 8;  fr = 7; a = @('bt=5', 'bw=16', 'bh=16', 'l=3', 't=2', 'r=4', 'b=1') },
  @{ n = 'dump_bt2_10b';      f = 'GRAY';   b = 10; fr = 7; tol = $true; a = @('bt=2', 'bw=16', 'bh=16') },
  @{ n = 'dump_bt3_wt1';      f = 'GRAY';   b = 8;  fr = 7; a = @('bt=3', 'bw=16', 'bh=16', 'wintype=1') },
  @{ n = 'dump_bt5_ilace';    f = 'GRAY';   b = 8;  fr = 7; a = @('bt=5', 'bw=16', 'bh=16', 'interlaced=1') },
  # sharpen only (bt=-1): FFT -> Sharpen on the raw spectrum -> inverse FFT
  @{ n = 'dump_bm1_sharpen';  f = 'GRAY';   b = 8;  a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2', 'sharpen=1') },
  @{ n = 'dump_bm1_dehalo';   f = 'GRAY';   b = 8;  a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2', 'dehalo=1') },
  @{ n = 'dump_bm1_both';     f = 'GRAY';   b = 8;  a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2', 'sharpen=1', 'dehalo=1') },
  @{ n = 'dump_bm1_off';      f = 'GRAY';   b = 8;  a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2') },
  @{ n = 'dump_bm1_yuv';      f = 'YUV420'; b = 8;  a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2', 'sharpen=1', 'planes=0,1') },
  @{ n = 'dump_bm1_10b';      f = 'GRAY';   b = 10; a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2', 'sharpen=1') },
  @{ n = 'dump_bm1_multi';    f = 'GRAY';   b = 8;  fr = 3; a = @('bt=-1', 'bw=16', 'bh=16', 'sigma=2', 'sharpen=1') },
  # Kalman (bt=0): sequential IIR state, frame 0 passthrough; no rev/shuf order
  @{ n = 'dump_kalman';       f = 'GRAY';   b = 8;  fr = 7; a = @('bt=0', 'bw=16', 'bh=16', 'sigma=2') },
  @{ n = 'dump_kalman_sh';    f = 'GRAY';   b = 8;  fr = 7; a = @('bt=0', 'bw=16', 'bh=16', 'sigma=2', 'sharpen=1') },
  @{ n = 'dump_kalman_kr3';   f = 'GRAY';   b = 8;  fr = 7; a = @('bt=0', 'bw=16', 'bh=16', 'sigma=2', 'kratio=3') },
  @{ n = 'dump_kalman_10b';   f = 'GRAY';   b = 10; fr = 7; a = @('bt=0', 'bw=16', 'bh=16', 'sigma=2') },
  @{ n = 'dump_kalman_yuv';   f = 'YUV420'; b = 8;  fr = 7; a = @('bt=0', 'bw=16', 'bh=16', 'sigma=2') },
  # pattern denoise (pfactor != 0): pattern estimated once from frame pframe
  @{ n = 'dump_pat_bt1';      f = 'GRAY';   b = 8;  fr = 7; a = @('bt=1', 'bw=16', 'bh=16', 'pfactor=1') },
  @{ n = 'dump_pat_bt1_sh';   f = 'GRAY';   b = 8;  fr = 7; a = @('bt=1', 'bw=16', 'bh=16', 'pfactor=1', 'sharpen=1', 'dehalo=1') },
  @{ n = 'dump_pat_bt3';      f = 'GRAY';   b = 8;  fr = 7; a = @('bt=3', 'bw=16', 'bh=16', 'pfactor=1') },
  @{ n = 'dump_pat_bt5';      f = 'GRAY';   b = 8;  fr = 7; a = @('bt=5', 'bw=16', 'bh=16', 'pfactor=1') },
  @{ n = 'dump_pat_kal';      f = 'GRAY';   b = 8;  fr = 7; a = @('bt=0', 'bw=16', 'bh=16', 'pfactor=1') },
  @{ n = 'dump_pat_px';       f = 'GRAY';   b = 8;  fr = 7; a = @('bt=1', 'bw=16', 'bh=16', 'pfactor=1', 'px=3', 'py=3') },
  @{ n = 'dump_pat_sig';      f = 'GRAY';   b = 8;  fr = 7; a = @('bt=1', 'bw=16', 'bh=16', 'sigma2=1', 'sigma3=3', 'sigma4=5') },
  # pframe=2: one pixel sits exactly on the 8-bit rounding boundary and the GPU
  # FFT differs from FFTW by ~1 ULP, so that pixel may land ±1 LSB (deterministic)
  @{ n = 'dump_pat_pframe2';  f = 'GRAY';   b = 8;  fr = 7; tol = $true; a = @('bt=3', 'bw=16', 'bh=16', 'pfactor=1', 'pframe=2') },
  @{ n = 'dump_pat_yuv';      f = 'YUV420'; b = 8;  fr = 7; a = @('bt=1', 'bw=16', 'bh=16', 'pfactor=1', 'planes=0,1') },
  @{ n = 'dump_pat_nodg';     f = 'GRAY';   b = 8;  fr = 7; a = @('bt=1', 'bw=16', 'bh=16', 'pfactor=1', 'degrid=0') }
)

$fail = 0
foreach ($c in $cases) {
  $dir = Join-Path $root "tests\$($c.n)"
  $out = Join-Path $dir 'plugin_out.raw'
  $fa = @((Join-Path $dir 'input.raw'), (Join-Path $dir 'out.raw'),
          '64', '64', "$($c.b)", $c.f, $out) + $c.a
  if ($c.fr) { $fa += "frames=$($c.fr)" }
  if ($c.n -eq 'dump_bt128') { $fa[2] = '192'; $fa[3] = '192' }
  # stderr du plugin (log VRAM) ne doit pas devenir une erreur terminante (PS 5.1)
  $ErrorActionPreference = 'Continue'
  $line = & $smoke @fa 2>$null | Select-String 'ndiff=' | Select-Object -First 1
  $ErrorActionPreference = 'Stop'
  "{0,-18} {1}" -f $c.n, (($line | Select-Object -First 1) -replace '.*ndiff=', 'ndiff=')
  if ($line -notmatch 'ndiff=(\d+) .*maxabs=([0-9.eE+-]+)') { $fail++; continue }
  $nd = [int]$Matches[1]; $mx = [double]$Matches[2]
  if ($c.tol) {
    # expected: at most 1 LSB (integer) or 1e-5 (float32)
    $lim = if ($c.b -eq 32) { 1e-5 } else { 1.0 }
    if ($mx -gt $lim) { $fail++ }
  } elseif ($nd -ne 0) { $fail++ }
}
"fixtures hors tolerance : $fail (attendu 0 ; 14b/16b/32f/chroma_f/bt2_10b/pat_pframe2 = ±1 LSB accepté)"
