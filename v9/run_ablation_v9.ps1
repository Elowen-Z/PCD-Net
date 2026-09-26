<#
.SYNOPSIS
    Train and evaluate the paper-aligned V9 routing and context ablations.

.DESCRIPTION
    Reuses the existing full V9 checkpoint, trains only the uniform-alpha
    no-routing variant, and evaluates both checkpoints on one paired STEAD test
    set.  It then performs the inference-only background-context ablation.
#>
param(
    [ValidateSet("quick", "full")]
    [string]$Profile = "full",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$EvalDevice = "auto",
    [int]$Seed = 0,
    [string]$Python = "D:\app\python\envs\EarthquakeDetection\python.exe",
    [string]$EventH5 = "E:/p_wave/data/chunk2.hdf5",
    [string]$TrainEventCsv = "E:/p_wave/data/chunk2.csv",
    [string]$EvalEventCsv = "v5/v7_seed0_val.csv",
    [string]$NoiseH5 = "E:/p_wave/data/chunk1.hdf5",
    [string]$NoiseCsv = "E:/p_wave/data/chunk1.csv",
    [string]$FullCheckpoint = "v9/experiments/snr_m15_p10_seed0/best_model_v9.pth",
    [string]$NoRoutingRoot = "",
    [string]$NoPrototypeRoot = "",
    [string]$WeightedConcatRoot = "",
    [string]$OutputRoot = "",
    [switch]$SkipTraining,
    [switch]$SkipRouting,
    [switch]$SkipPrototype,
    [switch]$SkipAttention,
    [switch]$SkipContext
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $PSScriptRoot
Set-Location $Workspace

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
foreach ($File in @($EventH5, $TrainEventCsv, $EvalEventCsv, $NoiseH5, $NoiseCsv, $FullCheckpoint)) {
    if (-not (Test-Path -LiteralPath $File)) { throw "Required file not found: $File" }
}

if ($Profile -eq "quick") {
    $Epochs = 2
    $TrainSamples = 400
    $ValSamples = 100
    $EvalSamples = 100
} else {
    $Epochs = 30
    $TrainSamples = 4000
    $ValSamples = 2000
    $EvalSamples = 2000
}
$ProfileSuffix = if ($Profile -eq "quick") { "_quick" } else { "" }
if (-not $NoRoutingRoot) {
    $NoRoutingRoot = "v9/experiments/ablation_no_routing_snr_m15_p10_seed$Seed$ProfileSuffix"
}
if (-not $NoPrototypeRoot) {
    $NoPrototypeRoot = "v9/experiments/ablation_no_prototype_snr_m15_p10_seed$Seed$ProfileSuffix"
}
if (-not $WeightedConcatRoot) {
    $WeightedConcatRoot = "v9/experiments/ablation_weighted_concat_snr_m15_p10_seed$Seed$ProfileSuffix"
}
if (-not $OutputRoot) {
    $OutputRoot = "v9/paper_experiments/ablation_v9_seed$Seed$ProfileSuffix"
}
$BatchSize = 4
$NoRoutingCheckpoint = Join-Path $NoRoutingRoot "best_model_v9.pth"
$NoPrototypeCheckpoint = Join-Path $NoPrototypeRoot "best_model_v9.pth"
$WeightedConcatCheckpoint = Join-Path $WeightedConcatRoot "best_model_v9.pth"
$EvaluationExtraArgs = @()
if ($Profile -eq "quick") {
    $EvaluationExtraArgs += "--allow_protocol_mismatch"
}

Write-Host ""
Write-Host ("=" * 78)
Write-Host "V9 ablation | profile=$Profile | seed=$Seed"
Write-Host "Full checkpoint : $FullCheckpoint"
Write-Host "No-routing root : $NoRoutingRoot"
Write-Host "Input SNR       : [-15, 10] dB (uniform in dB)"
Write-Host ("=" * 78)

function Invoke-V9AblationTraining([string]$Name, [string]$SaveDir, [string[]]$ExtraArgs, [string]$Checkpoint) {
    if ($SkipTraining -or (Test-Path -LiteralPath $Checkpoint)) {
        Write-Host "[train] skip $Name : $Checkpoint"
        return
    }
    Write-Host "[train] $Name"
    $TrainArgs = @(
        "-u", "-m", "v9.train_v9",
        "--dataset_type", "stead", "--seed", "$Seed",
        "--epochs", "$Epochs", "--batch_size", "$BatchSize",
        "--train_samples", "$TrainSamples", "--val_samples", "$ValSamples",
        "--snr_min_db=-15", "--snr_max_db=10", "--n_refine", "0",
        "--context_source", "pre_event", "--num_workers", "0",
        "--event_h5", $EventH5, "--event_csv", $TrainEventCsv,
        "--noise_h5", $NoiseH5, "--noise_csv", $NoiseCsv,
        "--cuda_safe", "--save_dir", $SaveDir
    )
    $TrainArgs += $ExtraArgs
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name training failed" }
}

if (-not $SkipRouting) {
    Invoke-V9AblationTraining -Name "w/o routing (uniform alpha)" -SaveDir $NoRoutingRoot -ExtraArgs @("--no_routing") -Checkpoint $NoRoutingCheckpoint
}
if (-not $SkipPrototype) {
    Invoke-V9AblationTraining -Name "w/o prototype memory" -SaveDir $NoPrototypeRoot -ExtraArgs @("--no_prototypes") -Checkpoint $NoPrototypeCheckpoint
}
if (-not $SkipAttention) {
    Invoke-V9AblationTraining -Name "weighted-sum + concat fusion" -SaveDir $WeightedConcatRoot -ExtraArgs @("--prototype_interaction", "weighted_concat") -Checkpoint $WeightedConcatCheckpoint
}
foreach ($Checkpoint in @($NoRoutingCheckpoint, $NoPrototypeCheckpoint, $WeightedConcatCheckpoint)) {
    if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Ablation checkpoint not found: $Checkpoint" }
}

Write-Host "[evaluate] paired routing ablation"
& $Python -u -m v9.evaluate_routing_ablation_v9 `
    --full_checkpoint $FullCheckpoint `
    --no_routing_checkpoint $NoRoutingCheckpoint `
    --no_prototype_checkpoint $NoPrototypeCheckpoint `
    --weighted_concat_checkpoint $WeightedConcatCheckpoint `
    --event_h5 $EventH5 `
    --event_csv $EvalEventCsv `
    --noise_h5 $NoiseH5 `
    --noise_csv $NoiseCsv `
    --snr_min_db=-15 `
    --snr_max_db=10 `
    --max_samples $EvalSamples `
    --batch_size $BatchSize `
    --seed $Seed `
    --output_dir (Join-Path $OutputRoot "routing") `
    --cuda_safe `
    --device $EvalDevice `
    @EvaluationExtraArgs
if ($LASTEXITCODE -ne 0) { throw "Routing ablation evaluation failed" }

if (-not $SkipContext) {
    Write-Host "[evaluate] background-context ablation"
    & $Python -u -m v9.evaluate_context_ablation_v9 `
        --checkpoint $FullCheckpoint `
        --event_h5 $EventH5 `
        --event_csv $EvalEventCsv `
        --noise_h5 $NoiseH5 `
        --noise_csv $NoiseCsv `
        --snr_min_db=-15 `
        --snr_max_db=10 `
        --max_samples $EvalSamples `
        --batch_size $BatchSize `
        --seed $Seed `
        --cuda_safe `
        --device $EvalDevice `
        --output_dir (Join-Path $OutputRoot "context")
    if ($LASTEXITCODE -ne 0) { throw "Context ablation evaluation failed" }
}

Write-Host ""
Write-Host "[done] V9 ablation outputs: $OutputRoot"
