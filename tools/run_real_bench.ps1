# Comparatif fft3dneocl (GPU) vs fft3dcl (GPU) vs moteur CPU neo-fft3d (fft3d_ref)
# sur video reelle : Sample.mkv, 2 sequences de 60 images (171 et 419), 5 presets.
# Mesures : ms/image (2e passage), pic RAM (working set), pic VRAM (delta nvidia-smi),
# determinisme (hash des 2 passages), qualite (bench_compare.py vs reference CPU).
# Le DLL neo-fft3d.dll (API3) ne se charge pas dans R81 : le cote CPU est mesure avec
# fft3d_ref.exe, le moteur neo_FFT3D identique compile localement.
# Usage: powershell -File tools\run_real_bench.ps1 [-Presets 1,2,3,4,5] [-Seqs 171,419]
param([int[]]$Presets = @(1..5), [int[]]$Seqs = @(171, 419))
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$smoke = Join-Path $root 'fft3dneocl\build\Release\plugin_smoke.exe'
$ref   = Join-Path $root 'reference\harness\build\Release\fft3d_ref.exe'
$cl    = 'C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\build\Release\fft3dcl.dll'
$py    = 'C:\Users\Bruno\AppData\Local\Programs\Python\Python312\python.exe'
$dir   = Join-Path $root 'tests\bench_real'
$tmp   = Join-Path $env:TEMP 'opencode'
$W = 3840; $H = 2160; $BITS = 10; $FR = 60
if (-not (Test-Path $cl)) { Write-Error "missing fft3dcl: $cl" }

# lance un exe, surveille RAM (working set) et VRAM GPU (delta nvidia-smi global)
function Run-Watched {
  param([string]$exe, [string[]]$argv)
  $argv = $argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
  $base = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
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
  @{ out = $out; ramMB = [math]::Round($peakRam / 1MB, 0); vramMB = $peakGpu; code = $p.ExitCode }
}

# deux executions : perf sur la 2e (GPU chaud), RAM pic = max, VRAM pic = min des deux,
# determinisme = hash du fichier de sortie compare entre les 2 passages
function Run-Twice {
  param([string]$exe, [string[]]$argv, [string]$hashFile)
  $r1 = Run-Watched $exe $argv
  $h1 = if ($hashFile -and (Test-Path $hashFile)) { (Get-FileHash -Algorithm MD5 $hashFile).Hash } else { '' }
  $r2 = Run-Watched $exe $argv
  $h2 = if ($hashFile -and (Test-Path $hashFile)) { (Get-FileHash -Algorithm MD5 $hashFile).Hash } else { '' }
  $r2.vramMB = [math]::Min($r1.vramMB, $r2.vramMB)
  $r2.ramMB  = [math]::Max($r1.ramMB, $r2.ramMB)
  $r2.det    = ($h1 -ne '' -and $h1 -eq $h2)
  $r2
}

function Get-Perf {
  param([string]$out, [string]$pattern)
  $m = [regex]::Match($out, "$pattern.*?([\d.]+) ms/frame")
  if ($m.Success) { return [double]$m.Groups[1].Value } else { return $null }
}

# presets : gpu = args plugin (neocl et fft3dcl partagent les noms communs),
# cpu = args fft3d_ref. planes: neocl = liste, fft3dcl = plane=4 (tous).
$presetList = @(
  @{ n = 'P1 leger';    gpu = @('sigma=1.5'); cpu = @('--sigma', '1.5'); bt5 = $false },
  @{ n = 'P2 moyen';    gpu = @('sigma=3.0'); cpu = @('--sigma', '3.0'); bt5 = $false },
  @{ n = 'P3 pousse';   gpu = @('sigma=4.0', 'sharpen=1.0'); cpu = @('--sigma', '4.0', '--sharpen', '1.0'); bt5 = $false },
  @{ n = 'P4 bt=5';     gpu = @('sigma=3.0', 'bt=5'); cpu = @('--sigma', '3.0', '--bt', '5'); bt5 = $true },
  @{ n = 'P5 2 passes'; gpu = @('sigma=2.5'); cpu = @('--sigma', '2.5'); bt5 = $false; two = $true }
)
$presetList = $presetList | Where-Object { $Presets -contains [int]($_.n.Substring(1, 1)) }
$seqs = $Seqs
$results = @()

foreach ($s in $seqs) {
  $seq = Join-Path $dir "seq_$s.raw"
  if (-not (Test-Path $seq)) { Write-Error "missing $seq (lancer bench_dump.py)" }
  foreach ($p in $presetList) {
    $pd = Join-Path $dir "seq$s"
    New-Item -ItemType Directory -Force -Path $pd | Out-Null
    ""
    "=== seq $s / $($p.n) ==="

    if ($p.two) {
      # ---- 2 passes : p1 luma+chroma, p2 luma seule ----
      # neocl
      $o1 = Join-Path $pd "$($p.n.Split(' ')[0])_neocl_p1.raw"; $o2 = Join-Path $pd "$($p.n.Split(' ')[0])_neocl_p2.raw"
      $a1 = @($seq, '-', $W, $H, $BITS, 'YUV420', $o1, 'planes=0,1,2') + $p.gpu + @('frames=60', 'time=1')
      $r1 = Run-Twice $smoke $a1 $o1
      $a2 = @($o1, '-', $W, $H, $BITS, 'YUV420', $o2, 'planes=0') + $p.gpu + @('frames=60', 'time=1')
      $r2 = Run-Twice $smoke $a2 $o2
      $m1 = Get-Perf $r1.out 'perf fft3dneocl'; $m2 = Get-Perf $r2.out 'perf fft3dneocl'
      $results += @{ seq = $s; preset = $p.n; filter = 'fft3dneocl'; ms = ($m1 + $m2); ms1 = $m1; ms2 = $m2
                     ram = [math]::Max($r1.ramMB, $r2.ramMB); vram = [math]::Max($r1.vramMB, $r2.vramMB)
                     vram1 = $r1.vramMB; vram2 = $r2.vramMB
                     det = ($r1.det -and $r2.det); out = $o2 }
      "neocl  : p1 $m1 + p2 $m2 ms/img   RAM $($results[-1].ram) MB  VRAM p1 $($r1.vramMB) / p2 $($r2.vramMB) MB  det=$($results[-1].det)"
      # fft3dcl
      $c1 = Join-Path $pd "$($p.n.Split(' ')[0])_cl_p1.raw"; $c2 = Join-Path $pd "$($p.n.Split(' ')[0])_cl_p2.raw"
      $b1 = @($seq, '-', $W, $H, $BITS, 'YUV420', 'unused.raw', 'noplugin=1', "ref=$cl",
              'refns=fft3dcl', 'reffn=FFT3DCL', 'device=1', 'plane=4') + $p.gpu + @('frames=60', 'time=1', "refout=$c1")
      $q1 = Run-Twice $smoke $b1 $c1
      $b2 = @($c1, '-', $W, $H, $BITS, 'YUV420', 'unused.raw', 'noplugin=1', "ref=$cl",
              'refns=fft3dcl', 'reffn=FFT3DCL', 'device=1', 'plane=0') + $p.gpu + @('frames=60', 'time=1', "refout=$c2")
      $q2 = Run-Twice $smoke $b2 $c2
      $n1 = Get-Perf $q1.out 'perf fft3dcl'; $n2 = Get-Perf $q2.out 'perf fft3dcl'
      $results += @{ seq = $s; preset = $p.n; filter = 'fft3dcl'; ms = ($n1 + $n2); ms1 = $n1; ms2 = $n2
                     ram = [math]::Max($q1.ramMB, $q2.ramMB); vram = [math]::Max($q1.vramMB, $q2.vramMB)
                     vram1 = $q1.vramMB; vram2 = $q2.vramMB
                     det = ($q1.det -and $q2.det); out = $c2 }
      "fft3dcl: p1 $n1 + p2 $n2 ms/img   RAM $($results[-1].ram) MB  VRAM p1 $($q1.vramMB) / p2 $($q2.vramMB) MB  det=$($results[-1].det)"
      # cpu
      $u1 = Join-Path $pd "$($p.n.Split(' ')[0])_cpu_p1.raw"; $u2 = Join-Path $pd "$($p.n.Split(' ')[0])_cpu_p2.raw"
      $d1 = @('--in', $seq, '--out', $u1, '--width', $W, '--height', $H, '--format', 'YUV420',
              '--bitdepth', $BITS, '--frames', $FR, '--planes', '0,1,2') + $p.cpu + @('--time')
      $w1 = Run-Watched $ref $d1
      $d2 = @('--in', $u1, '--out', $u2, '--width', $W, '--height', $H, '--format', 'YUV420',
              '--bitdepth', $BITS, '--frames', $FR, '--planes', '0') + $p.cpu + @('--time')
      $w2 = Run-Watched $ref $d2
      $k1 = Get-Perf $w1.out 'cpu engine'; $k2 = Get-Perf $w2.out 'cpu engine'
      $results += @{ seq = $s; preset = $p.n; filter = 'neo-fft3d (CPU)'; ms = ($k1 + $k2); ms1 = $k1; ms2 = $k2
                     ram = [math]::Max($w1.ramMB, $w2.ramMB); vram = $null; det = $true; out = $u2 }
      "CPU    : p1 $k1 + p2 $k2 ms/img   RAM $($results[-1].ram) MB"
      $cpuOut = $u2; $neoclOut = $o2; $clOut = $c2
    } else {
      # ---- passe unique ----
      $tag = $p.n.Split(' ')[0]
      $oN = Join-Path $pd "${tag}_neocl.raw"
      $aN = @($seq, '-', $W, $H, $BITS, 'YUV420', $oN, 'planes=0,1,2') + $p.gpu + @('frames=60', 'time=1')
      $rN = Run-Twice $smoke $aN $oN
      $mN = Get-Perf $rN.out 'perf fft3dneocl'
      $results += @{ seq = $s; preset = $p.n; filter = 'fft3dneocl'; ms = $mN; ram = $rN.ramMB
                     vram = $rN.vramMB; det = $rN.det; out = $oN }
      "neocl  : $mN ms/img   RAM $($rN.ramMB) MB  VRAM $($rN.vramMB) MB  det=$($rN.det)"

      if ($p.bt5) {
        "fft3dcl: bt=5 non supporte (bt valide : -1..4)"
        $clOut = $null
      } else {
        $oC = Join-Path $pd "${tag}_cl.raw"
        $aC = @($seq, '-', $W, $H, $BITS, 'YUV420', 'unused.raw', 'noplugin=1', "ref=$cl",
                'refns=fft3dcl', 'reffn=FFT3DCL', 'device=1', 'plane=4') + $p.gpu + @('frames=60', 'time=1', "refout=$oC")
        $rC = Run-Twice $smoke $aC $oC
        $mC = Get-Perf $rC.out 'perf fft3dcl'
        $results += @{ seq = $s; preset = $p.n; filter = 'fft3dcl'; ms = $mC; ram = $rC.ramMB
                       vram = $rC.vramMB; det = $rC.det; out = $oC }
        "fft3dcl: $mC ms/img   RAM $($rC.ramMB) MB  VRAM $($rC.vramMB) MB  det=$($rC.det)"
        $clOut = $oC
      }

      $oU = Join-Path $pd "${tag}_cpu.raw"
      $aU = @('--in', $seq, '--out', $oU, '--width', $W, '--height', $H, '--format', 'YUV420',
              '--bitdepth', $BITS, '--frames', $FR, '--planes', '0,1,2') + $p.cpu + @('--time')
      $rU = Run-Watched $ref $aU
      $mU = Get-Perf $rU.out 'cpu engine'
      $results += @{ seq = $s; preset = $p.n; filter = 'neo-fft3d (CPU)'; ms = $mU; ram = $rU.ramMB
                     vram = $null; det = $true; out = $oU }
      "CPU    : $mU ms/img   RAM $($rU.ramMB) MB"
      $cpuOut = $oU; $neoclOut = $oN
    }

    # ---- qualite : chaque sortie GPU vs reference CPU, puis neocl vs fft3dcl ----
    foreach ($pair in @(@('neocl', 'CPU', $neoclOut, $cpuOut), @('cl', 'CPU', $clOut, $cpuOut), @('neocl', 'cl', $neoclOut, $clOut))) {
      if (-not $pair[2] -or -not $pair[3] -or
          -not (Test-Path $pair[2]) -or -not (Test-Path $pair[3])) { continue }
      $j = & $py (Join-Path $root 'tools\bench_compare.py') $pair[2] $pair[3] $W $H $BITS $FR 1 1 | ConvertFrom-Json
      $results += @{ seq = $s; preset = $p.n; filter = "qualite $($pair[0]) vs $($pair[1])"
                     pct = [math]::Round($j.pct, 3); maxabs = $j.maxabs; mae = [math]::Round($j.mae, 4)
                     psnr = $j.psnr; identical = $j.identical }
      "  qualite $($pair[0]) vs $($pair[1]) : diff $($j.pct)%  maxabs $($j.maxabs)  PSNR $($j.psnr)"
    }
  }
}

$json = $results | ConvertTo-Json -Depth 4
$json | Set-Content -Encoding UTF8 (Join-Path $dir 'results.json')
""
"Resultats ecrits dans tests\bench_real\results.json"
