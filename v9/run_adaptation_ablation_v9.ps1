<#
.SYNOPSIS
    Run V9 target-domain adaptation-scope and label-efficiency experiments.

.DESCRIPTION
    Trains context-encoder-only, environment-branch, and full fine-tuning
    variants from the same source checkpoint.  Each target sample budget uses
    the same target train/validation split seed and is evaluated on the supplied
    held-out target test CSV.
#>
param(
    [string]$Python = "D:\app\python\envs\EarthquakeDetection\python.exe",
    [ValidateSet("mining", "stead")]
    [string]$DatasetType = "mining",
    [string]$EventH5 = "D:/X/part2/data/LN_mining.hdf5",
    [string]$TrainEventCsv = "D:/X/part2/data/LN_mining.csv",
    [string]$EvalEventCsv = "",
    [string]$NoiseH5 = "E:/p_wave/data/chunk1.hdf5",
    [string]$NoiseCsv = "E:/p_wave/data/chunk1.csv",
    [string]$SourceCheckpoint = "v9/experiments/snr_m15_p10_seed0/best_model_v9.pth",
    [int[]]$TrainSampleCounts = @(500),
    [double[]]$TrainPercentages = @(),
    [int]$ValSamples = 500,
    [int]$Epochs = 10,
    [int]$BatchSize = 4,
    [int]$Seed = 0,
    [int]$EvalSamples = 2000,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $PSScriptRoot
Set-Location $Workspace
if (-not $OutputRoot) {
    $OutputRoot = "v9/paper_experiments/adaptation_ablation_v9_seed${Seed}_${DatasetType}"
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
foreach ($File in @($EventH5, $TrainEventCsv, $NoiseH5, $NoiseCsv, $SourceCheckpoint)) {
    if (-not (Test-Path -LiteralPath $File)) { throw "Required file not found: $File" }
}
if (-not $EvalEventCsv) {
    $SplitDir = Join-Path $OutputRoot "splits"
    & $Python -u -m v9.prepare_adaptation_split_v9 `
        --input_csv $TrainEventCsv `
        --output_dir $SplitDir `
        --seed $Seed `
        --test_fraction 0.2 `
        --dataset_type $DatasetType
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare target train/test split" }
    $TrainEventCsv = Join-Path $SplitDir "target_train_pool_seed$Seed.csv"
    $EvalEventCsv = Join-Path $SplitDir "target_test_seed$Seed.csv"
} elseif (-not (Test-Path -LiteralPath $EvalEventCsv)) {
    throw "Held-out target test CSV not found: $EvalEventCsv"
}

$Strategies = @(
    @{ Name = "context_encoder"; Label = "Context encoder only" },
    @{ Name = "environment"; Label = "Environment branch (encoder + memory + routing)" },
    @{ Name = "decoder_only"; Label = "Waveform decoder only" },
    @{ Name = "environment_decoder"; Label = "Env + Decoder" },
    @{ Name = "full"; Label = "Full fine-tuning" }
)
$TargetPoolRows = (Import-Csv -LiteralPath $TrainEventCsv | Measure-Object).Count
$ValidationRows = [math]::Round($TargetPoolRows * 0.1)
$TrainingPoolSize = $TargetPoolRows - $ValidationRows
$BudgetRuns = @()
if ($TrainPercentages.Count -gt 0) {
    foreach ($Percentage in $TrainPercentages) {
        if ($Percentage -le 0 -or $Percentage -gt 100) { throw "TrainPercentages must be in (0, 100]." }
        $Budget = [math]::Max($BatchSize, [math]::Ceiling($TrainingPoolSize * $Percentage / 100.0))
        $Budget = [math]::Min($Budget, $TrainingPoolSize)
        $Label = ("p{0}" -f $Percentage).Replace('.', 'p')
        $BudgetRuns += [PSCustomObject]@{ TrainSamples = [int]$Budget; Percentage = $Percentage; Label = $Label }
    }
} else {
    foreach ($Budget in $TrainSampleCounts) {
        $BudgetRuns += [PSCustomObject]@{ TrainSamples = [int]$Budget; Percentage = $null; Label = "n$Budget" }
    }
}
$AllRows = @()
foreach ($BudgetRun in $BudgetRuns) {
    $Budget = $BudgetRun.TrainSamples
    if ($Budget -lt $BatchSize) { throw "Each training budget must be at least batch size ($BatchSize) for drop_last batches." }
    $BudgetRoot = Join-Path $OutputRoot $BudgetRun.Label
    $Checkpoints = @("--checkpoint", "Frozen=$SourceCheckpoint")
    foreach ($Strategy in $Strategies) {
        $SaveDir = Join-Path $BudgetRoot $Strategy.Name
        $Best = Join-Path $SaveDir "best_adapted_v9.pth"
        if (-not (Test-Path -LiteralPath $Best)) {
            Write-Host "[adapt] n=$Budget strategy=$($Strategy.Name)"
            & $Python -u -m v9.transfer_v9 `
                --checkpoint $SourceCheckpoint `
                --dataset_type $DatasetType `
                --strategy $Strategy.Name `
                --event_h5 $EventH5 `
                --event_csv $TrainEventCsv `
                --noise_h5 $NoiseH5 `
                --noise_csv $NoiseCsv `
                --context_source pre_event `
                --epochs $Epochs `
                --train_samples $Budget `
                --val_samples $ValSamples `
                --batch_size $BatchSize `
                --seed $Seed `
                --save_dir $SaveDir
            if ($LASTEXITCODE -ne 0) { throw "Adaptation failed: n=$Budget strategy=$($Strategy.Name)" }
        } else {
            Write-Host "[adapt] reuse $Best"
        }
        $Checkpoints += @("--checkpoint", "$($Strategy.Label)=$Best")
    }
    $EvalDir = Join-Path $BudgetRoot "test"
    Write-Host "[test] n=$Budget held-out samples=$EvalSamples"
    & $Python -u -m v9.evaluate_adaptation_v9 `
        @Checkpoints `
        --event_h5 $EventH5 `
        --event_csv $EvalEventCsv `
        --noise_h5 $NoiseH5 `
        --noise_csv $NoiseCsv `
        --context_source pre_event `
        --snr_min_db=-15 `
        --snr_max_db=10 `
        --max_samples $EvalSamples `
        --batch_size $BatchSize `
        --seed $Seed `
        --output_dir $EvalDir
    if ($LASTEXITCODE -ne 0) { throw "Adaptation evaluation failed: n=$Budget" }
    foreach ($Row in (Import-Csv (Join-Path $EvalDir "adaptation_ablation_table.csv"))) {
        $AllRows += [PSCustomObject]@{
            TrainPercentage = $BudgetRun.Percentage
            TrainSamples = $Budget
            Strategy = $Row.Strategy
            N = $Row.N
            DeltaSNR_dB = $Row.DeltaSNR_dB
            CC = $Row.CC
            RMSE = $Row.RMSE
            ST_MAE = $Row.ST_MAE
        }
    }
}
$AllRows | Export-Csv (Join-Path $OutputRoot "adaptation_label_efficiency.csv") -NoTypeInformation -Encoding UTF8
Write-Host "[saved] $(Join-Path $OutputRoot 'adaptation_label_efficiency.csv')"
