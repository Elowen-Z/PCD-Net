param(
    [ValidateSet("quick", "full")]
    [string]$Profile = "quick",
    [int]$Seed = 0,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $PSScriptRoot
Set-Location $Workspace

if ($Profile -eq "quick") {
    $Epochs = 2
    $TrainSamples = 400
    $ValSamples = 100
    $BatchSize = 4
} else {
    $Epochs = 20
    $TrainSamples = 0
    $ValSamples = 2000
    $BatchSize = 4
}

$Common = @(
    "-m", "v9.train_v9",
    "--dataset_type", "stead",
    "--seed", "$Seed",
    "--epochs", "$Epochs",
    "--train_samples", "$TrainSamples",
    "--val_samples", "$ValSamples",
    "--batch_size", "$BatchSize",
    "--n_refine", "0",
    "--context_source", "pre_event",
    "--num_workers", "0"
)

Write-Host "[Table 1] Full V9 soft-routing model"
& $Python @Common --save_dir "v9/experiments/table1_seed$Seed"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[Table 2] Routing ablation: uniform alpha"
& $Python @Common --no_routing --save_dir "v9/experiments/table2_no_routing_seed$Seed"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Table 3 uses v9.transfer_v9 after providing target-domain paths."
