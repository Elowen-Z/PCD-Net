<#
.SYNOPSIS
    Unified input-SNR controlled comparison for PCD-Net V9.

.DESCRIPTION
    Runs Bandpass / Wavelet / DeepDenoiser / DPRNN / Restormer1D /
    ProtoCondTCN / PCD-Net on ONE shared STEAD hold-out split with a fixed
    input-SNR interval (default -15 ... 10 dB, uniform in dB).

    Every method is fed the identical noisy waveform and the identical
    conditioning window for every index, and the runner verifies this by
    comparing the per-method input-SNR vector.  ``--context_source pre_event``
    matches the protocol PCD-Net V9 and ProtoCondTCN are trained with.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File v9/run_v9_comparison.ps1 -Profile smoke
    powershell -ExecutionPolicy Bypass -File v9/run_v9_comparison.ps1 -Profile full
    powershell -ExecutionPolicy Bypass -File v9/run_v9_comparison.ps1 -Profile full `
        -V9Checkpoint v9/experiments/snr_m15_p10_seed0_18k/best_model_v9.pth
#>
param(
    [ValidateSet("smoke", "quick", "full")]
    [string]$Profile = "full",
    [int]$Seed = 0,
    [string]$Methods = "PCD-Net,ProtoCondTCN,DPRNN,Restormer1D,DeepDenoiser,Wavelet,Bandpass",
    [double]$SnrMinDb = -15.0,
    [double]$SnrMaxDb = 10.0,
    [string]$SnrGroups = "",
    [string]$EventH5 = "E:/p_wave/data/chunk2.hdf5",
    [string]$EventCsv = "v5/v7_seed0_val.csv",
    [string]$NoiseH5 = "E:/p_wave/data/chunk1.hdf5",
    [string]$NoiseCsv = "E:/p_wave/data/chunk1.csv",
    [string]$V9Checkpoint = "v9/experiments/snr_m15_p10_seed0/best_model_v9.pth",
    [string]$ProtoCondCkpt = "v7/retrain_snr_m15_p10/ProtoCondTCN_seed0/best_proto_cond_tcn.pth",
    [string]$DeepDenoiserCkpt = "v3/baeslines/checkpoints_snr_m15_p10/deep_denoiser/best_model.pth",
    [string]$DprnnCkpt = "v3/baeslines/checkpoints_snr_m15_p10/dprnn/best_model.pth",
    [string]$RestormerCkpt = "v3/baeslines/checkpoints_snr_m15_p10/restormer1d/best_model.pth",
    [string]$SaveDir = "v9/paper_experiments/model_comparison_stead_snr_m15_p10",
    [switch]$CudaSafe,
    [switch]$Resume,
    [string]$Python = "D:\app\python\envs\EarthquakeDetection\python.exe"
)

$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $PSScriptRoot
Set-Location $Workspace

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "[warn] $Python not found; falling back to 'python' from PATH"
    $Python = "python"
}

switch ($Profile) {
    "smoke" { $MaxSamples = 24;  $BatchSize = 2 }
    "quick" { $MaxSamples = 200; $BatchSize = 2 }
    "full"  { $MaxSamples = 2000; $BatchSize = 4 }
}

$Arguments = @(
    "-u", "-m", "v9.compare_v9",
    "--event_h5", $EventH5,
    "--event_csv", $EventCsv,
    "--noise_h5", $NoiseH5,
    "--noise_csv", $NoiseCsv,
    "--context_source", "pre_event",
    "--snr_min_db=$SnrMinDb",
    "--snr_max_db=$SnrMaxDb",
    "--max_samples", "$MaxSamples",
    "--batch_size", "$BatchSize",
    "--seed", "$Seed",
    "--methods", $Methods,
    "--ckpt_v9", $V9Checkpoint,
    "--ckpt_proto_cond_tcn", $ProtoCondCkpt,
    "--ckpt_deepdenoiser", $DeepDenoiserCkpt,
    "--ckpt_dprnn", $DprnnCkpt,
    "--ckpt_restormer1d", $RestormerCkpt,
    "--save_dir", $SaveDir
)

# The value starts with '-', so the '--flag=value' form is required.
if ($SnrGroups -ne "") { $Arguments += "--snr_groups=$SnrGroups" }
if ($CudaSafe) { $Arguments += "--cuda_safe" }
if ($Resume)   { $Arguments += "--resume" }
if ($Profile -eq "smoke") { $Arguments += "--allow_missing" }

Write-Host ""
Write-Host "=" * 80
Write-Host "  PCD-Net V9 comparison | profile=$Profile samples=$MaxSamples batch=$BatchSize"
Write-Host "=" * 80
Write-Host "  python : $Python"
Write-Host "  output : $SaveDir"
Write-Host ""

& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$PlotDir = Join-Path $SaveDir "plots"
New-Item -ItemType Directory -Force -Path $PlotDir | Out-Null

Write-Host ""
Write-Host "[figures] input-SNR group gain"
& $Python -m v7.plot_snr_group_gain_comparison `
    --input (Join-Path $SaveDir "per_group_compare.csv") `
    --out_dir $PlotDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "[figures] full model comparison"
& $Python -m v7.plot_full_model_comparison_v7 `
    --csv (Join-Path $SaveDir "paper_model_comparison_table.csv") `
    --out_dir $PlotDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "[done]"
Write-Host "  Table 2 : $SaveDir\global_compare.csv"
Write-Host "  Table 3 : $SaveDir\per_group_compare.csv"
Write-Host "  LaTeX   : $SaveDir\paper_model_comparison_table.tex"
Write-Host "  Audit   : $SaveDir\test_set_audit.json"
Write-Host "  Figures : $PlotDir"
