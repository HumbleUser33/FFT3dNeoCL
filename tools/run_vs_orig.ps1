# Ecart a l'image d'origine : seq_*.raw vs chaque sortie denoisee (memes cas que
# run_real_bench.ps1). PSNR/MAE vs original = agressivite du denoising.
# Usage: powershell -File tools\run_vs_orig.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py  = 'C:\Users\Bruno\AppData\Local\Programs\Python\Python312\python.exe'
$cmp = Join-Path $root 'tools\bench_compare.py'
$dir = Join-Path $root 'tests\bench_real'
$W = 3840; $H = 2160; $BITS = 10; $FR = 60

$cases = @(
  @('P1', 'fft3dneocl', 'P1_neocl.raw'),   @('P1', 'neo-fft3d CPU', 'P1_cpu.raw'),   @('P1', 'fft3dcl', 'P1_cl.raw'),
  @('P2', 'fft3dneocl', 'P2_neocl.raw'),   @('P2', 'neo-fft3d CPU', 'P2_cpu.raw'),   @('P2', 'fft3dcl', 'P2_cl.raw'),
  @('P3', 'fft3dneocl', 'P3_neocl.raw'),   @('P3', 'neo-fft3d CPU', 'P3_cpu.raw'),   @('P3', 'fft3dcl', 'P3_cl.raw'),
  @('P4', 'fft3dneocl', 'P4_neocl.raw'),   @('P4', 'neo-fft3d CPU', 'P4_cpu.raw'),
  @('P5', 'fft3dneocl', 'P5_neocl_p2.raw'), @('P5', 'neo-fft3d CPU', 'P5_cpu_p2.raw'), @('P5', 'fft3dcl', 'P5_cl_p2.raw')
)
$res = @()
foreach ($s in @(171, 419)) {
  $orig = Join-Path $dir "seq_$s.raw"
  foreach ($c in $cases) {
    $out = Join-Path $dir "seq$s\$($c[2])"
    $j = & $py $cmp $orig $out $W $H $BITS $FR 1 1 | ConvertFrom-Json
    $res += @{ seq = $s; preset = $c[0]; filter = $c[1]; pct = [math]::Round($j.pct, 2)
               mae = [math]::Round($j.mae, 3); maxabs = $j.maxabs; psnr = $j.psnr
               y_psnr = [math]::Round($j.planes[0].psnr, 2)
               u_psnr = [math]::Round($j.planes[1].psnr, 2); v_psnr = [math]::Round($j.planes[2].psnr, 2) }
    "seq $s  $($c[0])  $($c[1].PadRight(13))  modifie $($j.pct)%  MAE $([math]::Round($j.mae,3))  maxabs $($j.maxabs)  PSNR $($j.psnr)  (Y $($j.planes[0].psnr) U $($j.planes[1].psnr) V $($j.planes[2].psnr))"
  }
  ""
}
$res | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $dir 'vs_orig.json')
"Resultats ecrits dans tests\bench_real\vs_orig.json"
