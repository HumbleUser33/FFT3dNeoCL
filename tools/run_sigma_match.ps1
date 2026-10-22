# Balayage sigma (puis sharpen) pour trouver l'equivalence visuelle fft3dneocl
# vs fft3dcl a blocs identiques (bw=bh=32, ow=oh=16), bt=3 et bt=4.
# 10 images de seq_171. Sorties brutes dans tests\bench_real\match\.
# Phase A (sigma, sharpen=0) : powershell -File tools\run_sigma_match.ps1 -Phase A
# Phase B (sharpen)          : powershell -File tools\run_sigma_match.ps1 -Phase B -NeoSigma 3 -ClSigma 2.5
param([string]$Phase = 'A', [double]$NeoSigma = 3.0, [double]$ClSigma = 2.5, [int[]]$Bt = @(3, 4))
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$smoke = Join-Path $root 'fft3dneocl\build\Release\plugin_smoke.exe'
$cl    = 'C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\build\Release\fft3dcl.dll'
$dir   = Join-Path $root 'tests\bench_real\match'
$seq   = Join-Path $root 'tests\bench_real\seq_171.raw'
$W = 3840; $H = 2160; $FR = 10
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$common = @('bw=32', 'bh=32', 'ow=16', 'oh=16')
function Run-Neo { param($bt, $sigma, $sh, $out)
  $a = @($seq, '-', $W, $H, 10, 'YUV420', $out, 'planes=0,1,2') + $common +
       @("bt=$bt", "sigma=$sigma", "sharpen=$sh", "frames=$FR")
  & $smoke @a | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "neocl bt=$bt sigma=$sigma sharpen=$sh a echoue" }
}
function Run-Cl { param($bt, $sigma, $sh, $out)
  $a = @($seq, '-', $W, $H, 10, 'YUV420', 'unused.raw', 'noplugin=1', "ref=$cl",
         'refns=fft3dcl', 'reffn=FFT3DCL', 'device=1', 'plane=4') + $common +
       @("bt=$bt", "sigma=$sigma", "sharpen=$sh", "frames=$FR", "refout=$out")
  & $smoke @a | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "fft3dcl bt=$bt sigma=$sigma sharpen=$sh a echoue" }
}

if ($Phase -eq 'A') {
  $neoSig = @(1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
  $clSig  = @(1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
  foreach ($bt in @(3, 4)) {
    foreach ($s in $neoSig) { $o = Join-Path $dir "bt${bt}_neo_s$s.raw"; Run-Neo $bt $s 0 $o; "OK neo bt=$bt sigma=$s" }
    foreach ($s in $clSig)  { $o = Join-Path $dir "bt${bt}_cl_s$s.raw";  Run-Cl  $bt $s 0 $o; "OK cl  bt=$bt sigma=$s" }
  }
} else {
  $sh = @(0.25, 0.5, 0.75, 1.0)
  foreach ($bt in $Bt) {
    foreach ($a in $sh) { $o = Join-Path $dir "bt${bt}_neo_s${NeoSigma}_k$a.raw"; Run-Neo $bt $NeoSigma $a $o; "OK neo bt=$bt sigma=$NeoSigma sharpen=$a" }
    foreach ($b in $sh) { $o = Join-Path $dir "bt${bt}_cl_s${ClSigma}_k$b.raw";  Run-Cl  $bt $ClSigma  $b $o; "OK cl  bt=$bt sigma=$ClSigma sharpen=$b" }
  }
}
"Termine ($Phase)"
