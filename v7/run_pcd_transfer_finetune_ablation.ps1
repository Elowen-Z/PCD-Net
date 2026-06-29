param(
    [ValidateSet("mining", "nonnat", "both")]
    [string]$Target = "both",
    [int]$Stage2Epochs = 5,
    [int]$Stage3Epochs = 15,
    [int]$BatchSize = 2,
    [int]$NumWorkers = 0,
    [string]$Strategies = "noise_encoder,signal_encoder,signal_decoder,prototype_feedback,pcd_adaptation,full"
)

$ErrorActionPreference = "Stop"
$Python = "D:\app\anaconda\anaconda\envs\EarthquakeDetection\python.exe"
$Workspace = "D:\X\denoise\part1"
$SourceCheckpoint = Join-Path $Workspace "v7\checkpoints_feedback_stead_seed0\best_model_v7.pth"

Set-Location $Workspace

$Targets = if ($Target -eq "both") {
    @("mining", "nonnat")
}
else {
    @($Target)
}

foreach ($CurrentTarget in $Targets) {
    & $Python -u -m v7.run_transfer_comparison_v7 `
        --target $CurrentTarget `
        --suite freeze `
        --source_ckpt $SourceCheckpoint `
        --stage2_epochs $Stage2Epochs `
        --stage3_epochs $Stage3Epochs `
        --batch_size $BatchSize `
        --num_workers $NumWorkers `
        --print_every 20 `
        --freeze_strategies $Strategies

    if ($LASTEXITCODE -ne 0) {
        throw "PCD-Net fine-tuning ablation failed for $CurrentTarget"
    }
}
