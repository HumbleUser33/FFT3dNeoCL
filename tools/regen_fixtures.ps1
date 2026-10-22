# Regenerates the input.raw + out.raw (CPU reference) of every dump_* fixture
# used by run_plugin_matrix.ps1. Inputs come from the recorded seed in
# <fixture>\input.raw.meta.txt; the CPU reference comes from fft3d_ref.exe with
# the same filter arguments the matrix passes to the plugin. Deterministic:
# same seed + same engine = byte-identical files.
# Usage: powershell -NoProfile -File tools\regen_fixtures.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$ref = Join-Path $root 'reference\harness\build\Release\fft3d_ref.exe'
if (-not (Test-Path $ref)) { Write-Error "missing $ref (build reference/harness first)" }

# same table as run_plugin_matrix.ps1: fixture -> harness filter args
$cases = @(
  @{ n = 'dump_bt1';        a = @('--bt','1','--bw','16','--bh','16') },
  @{ n = 'dump_bt1_10b';    a = @('--bt','1','--bw','16','--bh','16') },
  @{ n = 'dump_bt1_12b';    a = @('--bt','1','--bw','16','--bh','16') },
  @{ n = 'dump_bt1_14b';    a = @('--bt','1','--bw','16','--bh','16') },
  @{ n = 'dump_bt1_16b';    a = @('--bt','1','--bw','16','--bh','16') },
  @{ n = 'dump_bt1_32f';    a = @('--bt','1','--bw','16','--bh','16') },
  @{ n = 'dump_bt32';       a = @('--bt','1','--bw','32','--bh','32') },
  @{ n = 'dump_bt64';       a = @('--bt','1','--bw','64','--bh','64') },
  @{ n = 'dump_bt128';      a = @('--bt','1','--bw','128','--bh','128') },
  @{ n = 'dump_wt1';        a = @('--bt','1','--bw','16','--bh','16','--wintype','1') },
  @{ n = 'dump_wt2';        a = @('--bt','1','--bw','16','--bh','16','--wintype','2') },
  @{ n = 'dump_ilace';      a = @('--bt','1','--bw','16','--bh','16','--interlaced','1') },
  @{ n = 'dump_crop';       a = @('--bt','1','--bw','16','--bh','16','--l','3','--t','2','--r','5','--b','4') },
  @{ n = 'dump_chroma';     a = @('--bt','1','--bw','16','--bh','16','--planes','0,1') },
  @{ n = 'dump_chroma_f';   a = @('--bt','1','--bw','16','--bh','16','--planes','0,1') },
  @{ n = 'dump_yuv422';     a = @('--bt','1','--bw','16','--bh','16','--planes','0,1') },
  @{ n = 'dump_yuv444';     a = @('--bt','1','--bw','16','--bh','16','--planes','0,1') },
  @{ n = 'dump_yuv_p0only'; a = @('--bt','1','--bw','16','--bh','16','--planes','0') },
  @{ n = 'dump_bt2';        a = @('--bt','2','--bw','16','--bh','16') },
  @{ n = 'dump_bt3';        a = @('--bt','3','--bw','16','--bh','16') },
  @{ n = 'dump_bt4';        a = @('--bt','4','--bw','16','--bh','16') },
  @{ n = 'dump_bt5';        a = @('--bt','5','--bw','16','--bh','16') },
  @{ n = 'dump_bt3_bw32';   a = @('--bt','3','--bw','32','--bh','32') },
  @{ n = 'dump_bt3_yuv';    a = @('--bt','3','--bw','16','--bh','16') },
  @{ n = 'dump_bt4_sharpen';a = @('--bt','4','--bw','16','--bh','16','--sharpen','1','--dehalo','1') },
  @{ n = 'dump_bt5_crop';   a = @('--bt','5','--bw','16','--bh','16','--l','3','--t','2','--r','4','--b','1') },
  @{ n = 'dump_bt2_10b';    a = @('--bt','2','--bw','16','--bh','16') },
  @{ n = 'dump_bt3_wt1';    a = @('--bt','3','--bw','16','--bh','16','--wintype','1') },
  @{ n = 'dump_bt5_ilace';  a = @('--bt','5','--bw','16','--bh','16','--interlaced','1') },
  @{ n = 'dump_bm1_sharpen';a = @('--bt','-1','--bw','16','--bh','16','--sigma','2','--sharpen','1') },
  @{ n = 'dump_bm1_dehalo'; a = @('--bt','-1','--bw','16','--bh','16','--sigma','2','--dehalo','1') },
  @{ n = 'dump_bm1_both';   a = @('--bt','-1','--bw','16','--bh','16','--sigma','2','--sharpen','1','--dehalo','1') },
  @{ n = 'dump_bm1_off';    a = @('--bt','-1','--bw','16','--bh','16','--sigma','2') },
  @{ n = 'dump_bm1_yuv';    a = @('--bt','-1','--bw','16','--bh','16','--sigma','2','--sharpen','1','--planes','0,1') },
  @{ n = 'dump_bm1_10b';    a = @('--bt','-1','--bw','16','--bh','16','--sigma','2','--sharpen','1') },
  @{ n = 'dump_bm1_multi';  a = @('--bt','-1','--bw','16','--bh','16','--sigma','2','--sharpen','1') },
  @{ n = 'dump_kalman';     a = @('--bt','0','--bw','16','--bh','16','--sigma','2') },
  @{ n = 'dump_kalman_sh';  a = @('--bt','0','--bw','16','--bh','16','--sigma','2','--sharpen','1') },
  @{ n = 'dump_kalman_kr3'; a = @('--bt','0','--bw','16','--bh','16','--sigma','2','--kratio','3') },
  @{ n = 'dump_kalman_10b'; a = @('--bt','0','--bw','16','--bh','16','--sigma','2') },
  @{ n = 'dump_kalman_yuv'; a = @('--bt','0','--bw','16','--bh','16','--sigma','2') },
  @{ n = 'dump_pat_bt1';    a = @('--bt','1','--bw','16','--bh','16','--pfactor','1') },
  @{ n = 'dump_pat_bt1_sh'; a = @('--bt','1','--bw','16','--bh','16','--pfactor','1','--sharpen','1','--dehalo','1') },
  @{ n = 'dump_pat_bt3';    a = @('--bt','3','--bw','16','--bh','16','--pfactor','1') },
  @{ n = 'dump_pat_bt5';    a = @('--bt','5','--bw','16','--bh','16','--pfactor','1') },
  @{ n = 'dump_pat_kal';    a = @('--bt','0','--bw','16','--bh','16','--pfactor','1') },
  @{ n = 'dump_pat_px';     a = @('--bt','1','--bw','16','--bh','16','--pfactor','1','--px','3','--py','3') },
  @{ n = 'dump_pat_sig';    a = @('--bt','1','--bw','16','--bh','16','--sigma2','1','--sigma3','3','--sigma4','5') },
  @{ n = 'dump_pat_pframe2';a = @('--bt','3','--bw','16','--bh','16','--pfactor','1','--pframe','2') },
  @{ n = 'dump_pat_yuv';    a = @('--bt','1','--bw','16','--bh','16','--pfactor','1','--planes','0,1') },
  @{ n = 'dump_pat_nodg';   a = @('--bt','1','--bw','16','--bh','16','--pfactor','1','--degrid','0') }
)

function Get-Meta([string]$file) {
  $m = @{}
  foreach ($l in Get-Content $file) { $k, $v = $l -split '=', 2; $m[$k] = $v }
  return $m
}

foreach ($c in $cases) {
  $dir = Join-Path $root "tests\$($c.n)"
  $metaFile = Join-Path $dir 'input.raw.meta.txt'
  if (-not (Test-Path $metaFile)) { Write-Error "missing $metaFile"; }
  $m = Get-Meta $metaFile
  $fmt = if ([int]$m['planes'] -eq 1) { 'GRAY' }
         elseif ([int]$m['planes'] -eq 3 -and [int]$m['ssw'] -eq 1 -and [int]$m['ssh'] -eq 1) { 'YUV420' }
         elseif ([int]$m['planes'] -eq 3 -and [int]$m['ssw'] -eq 1 -and [int]$m['ssh'] -eq 0) { 'YUV422' }
         elseif ([int]$m['planes'] -eq 3 -and [int]$m['ssw'] -eq 0 -and [int]$m['ssh'] -eq 0) { 'YUV444' }
         else { Write-Error "$($c.n): unknown plane/ssw/ssh combo"; 'GRAY' }
  $geo = @('--width', $m['width'], '--height', $m['height'], '--format', $fmt,
           '--bitdepth', $m['bitdepth'], '--frames', $m['frames'], '--seed', $m['seed'])
  # 1) input (deterministic from the seed)
  & $ref --dump-input --out (Join-Path $dir 'input.raw') @geo | Out-Null
  # 2) CPU reference output (same filter args as the matrix passes to the plugin)
  & $ref --out (Join-Path $dir 'out.raw') @geo @($c.a) | Out-Null
  "regen $($c.n)"
}

'done'
