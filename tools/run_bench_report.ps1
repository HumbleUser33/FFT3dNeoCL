# Bilan qualite / performances / memoire : fft3dneocl (GPU) vs fft3dcl (GPU) vs neo-fft3d (CPU).
# Pour chaque cas : ms/image, pic de RAM du processus (DRAM), pic de VRAM du processus
# (nvidia-smi compute-apps, par PID), et ecart de qualite vs la reference CPU (out.raw).
# Usage: powershell -File tools\run_bench_report.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$smoke = Join-Path $root 'fft3dneocl\build\Release\plugin_smoke.exe'
$ref   = Join-Path $root 'reference\harness\build\Release\fft3d_ref.exe'
$cl    = 'C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\build\Release\fft3dcl.dll'
$tmp   = Join-Path $env:TEMP 'opencode'
if (-not (Test-Path $cl)) { Write-Error "missing fft3dcl: $cl" }

# lance un exe, surveille RAM (working set) et VRAM GPU (delta nvidia-smi global)
function Run-Watched {
  param([string]$exe, [string[]]$argv)
  $argv = $argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
  $base = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
  $base = [int]$base
  $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru `
        -RedirectStandardOutput "$tmp\bench_out.txt" -RedirectStandardError "$tmp\bench_err.txt"
  $peakRam = 0; $peakGpu = 0
  while (-not $p.HasExited) {
    $p.Refresh()
    if ($p.WorkingSet64 -gt $peakRam) { $peakRam = [int64]$p.WorkingSet64 }
    $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    $d = $used - $base
    if ($d -gt $peakGpu) { $peakGpu = $d }
    Start-Sleep -Milliseconds 200
  }
  $out = ''
  if (Test-Path "$tmp\bench_out.txt") { $out += (Get-Content "$tmp\bench_out.txt" -Raw) }
  if (Test-Path "$tmp\bench_err.txt") { $out += "`n" + (Get-Content "$tmp\bench_err.txt" -Raw) }
  @{ out = $out; ramMB = [math]::Round($peakRam / 1MB, 0); vramMB = $peakGpu }
}

# deux executions : perf sur la 2e (GPU chaud), RAM pic = max, VRAM pic = min des deux
# pics (le delta global nvidia-smi est bruité par le bureau : le min approche le vrai pic)
function Run-Twice {
  param([string]$exe, [string[]]$argv)
  $r1 = Run-Watched $exe $argv
  $r2 = Run-Watched $exe $argv
  $r2.vramMB = [math]::Min($r1.vramMB, $r2.vramMB)
  $r2.ramMB = [math]::Max($r1.ramMB, $r2.ramMB)
  $r2
}

$cases = @(
  @{ n = '720p bt=1';  d = 'bench_720p_bt1_cpu';  w = 1280; h = 720;  fr = 30; bt = 1 },
  @{ n = '720p bt=3';  d = 'bench_720p_bt3_cpu';  w = 1280; h = 720;  fr = 30; bt = 3 },
  @{ n = '720p bt=5';  d = 'bench_720p_bt5_cpu';  w = 1280; h = 720;  fr = 30; bt = 5 },
  @{ n = '1080p bt=3'; d = 'bench_1080p_bt3_cpu'; w = 1920; h = 1080; fr = 30; bt = 3 }
)

foreach ($c in $cases) {
  $dir = Join-Path $root "tests\$($c.d)"
  $inp = Join-Path $dir 'input.raw'
  $outr = Join-Path $dir 'out.raw'
  ""
  "=== $($c.n) (GRAY8, bw=bh=32, sigma=2, $($c.fr) images) ==="

  # 1) neo-fft3d moteur CPU (fft3d_ref, mesure du moteur seul)
  $cpuArgs = @('--width', $c.w, '--height', $c.h, '--format', 'GRAY', '--bitdepth', 8,
               '--frames', $c.fr, '--seed', 5, '--sigma', 2, '--bw', 32, '--bh', 32,
               '--bt', $c.bt, '--time', '--out', "$tmp\cpu_out.raw")
  $r = Run-Watched $ref $cpuArgs
  "CPU  neo-fft3d : " + (($r.out -split "`n" | Where-Object { $_ -match 'cpu engine' }) -join '')
  "     RAM pic: $($r.ramMB) MB   VRAM: n/a"

  # 2) fft3dneocl (notre plugin GPU)
  $ourArgs = @($inp, $outr, $c.w, $c.h, 8, 'GRAY', (Join-Path $dir 'plugin_out.raw'),
               "bt=$($c.bt)", 'bw=32', 'bh=32', 'sigma=2', "frames=$($c.fr)", 'time=1')
  $r = Run-Twice $smoke $ourArgs
  "NEOCL GPU      : " + (($r.out -split "`n" | Where-Object { $_ -match 'perf fft3dneocl' }) -join '')
  "     qualite  : " + (($r.out -split "`n" | Where-Object { $_ -match 'total:' }) -join '')
  "     RAM pic: $($r.ramMB) MB   VRAM pic: $($r.vramMB) MB   " +
    (($r.out -split "`n" | Where-Object { $_ -match 'device buffers' }) -join '')

  # 3) fft3dcl (port OpenCL de FFT3DGPU), GPU force, sans notre plugin (noplugin)
  if ($c.bt -eq 5) {
    "fft3dcl GPU    : bt=5 non supporte (fft3dcl accepte bt=-1..4)"
    continue
  }
  $clArgs = $ourArgs + @("ref=$cl", 'refns=fft3dcl', 'reffn=FFT3DCL', 'device=1', 'noplugin=1',
                         "refout=$(Join-Path $dir 'fft3dcl_out.raw')")
  $r = Run-Twice $smoke $clArgs
  "fft3dcl GPU    : " + (($r.out -split "`n" | Where-Object { $_ -match 'perf fft3dcl' }) -join '')
  "     qualite  : " + (($r.out -split "`n" | Where-Object { $_ -match 'REF vs harness' }) -join '')
  "     RAM pic: $($r.ramMB) MB   VRAM pic: $($r.vramMB) MB"
}
""
"Memo: RAM pic = working set du processus (inclut les tampons du banc d'essai, identiques"
"d'un cote GPU a l'autre). VRAM pic = nvidia-smi compute-apps pour le PID du test (inclut"
"le contexte GPU et VapourSynth, communs aux deux plugins GPU)."
