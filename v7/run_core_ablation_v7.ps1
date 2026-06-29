param(
    [string]$DatasetType = "stead",
    [int]$Epochs = 30,
    [int]$Seed = 0,
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$TopM = 4,
    [int]$TrainSamples = 0,
    [int]$ValSamples = 0,
    [string]$PythonExe = "D:\app\anaconda\anaconda\envs\EarthquakeDetection\python.exe",
    [switch]$NoResume,
    [switch]$Fast,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = "python"
}

$variants = "no_proto_sparse,no_cross_feedback,no_quality_adaptive"
$argsList = @(
    ".\v7\run_ablation_v7.py",
    "--dataset_type", $DatasetType,
    "--epochs", "$Epochs",
    "--seed", "$Seed",
    "--top_m", "$TopM",
    "--variants", $variants,
    "--skip_completed",
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--val_batch_size", "$BatchSize",
    "--val_num_workers", "0",
    "--no_amp"
)

if (-not $Fast) {
    $argsList += "--cuda_safe"
    $argsList += "--no_pin_memory"
}

if ($TrainSamples -gt 0) {
    $argsList += "--train_samples"
    $argsList += "$TrainSamples"
}

if ($ValSamples -gt 0) {
    $argsList += "--val_samples"
    $argsList += "$ValSamples"
}

if (-not $NoResume) {
    $argsList += "--resume_existing"
}

if ($DryRun) {
    $argsList += "--dry_run"
}

Write-Host "Running core V7 ablation variants:"
Write-Host "  $variants"
Write-Host ""

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Core ablation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Core ablation finished. Results are under v7\exp_runs\ablation_v7."
Write-Host "After training, plot with:"
Write-Host "$PythonExe -m v7.plot_core_ablation_v7"
