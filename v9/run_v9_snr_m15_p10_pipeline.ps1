<#
.SYNOPSIS
    Full -15 ... 10 dB pipeline: shared split -> retrain -> comparison.

.DESCRIPTION
    Step 1  verifies/creates the shared STEAD hold-out split (seed 0, 10%).
    Step 2  retrains PCD-Net V9 at the target input-SNR interval (dB uniform).
    Step 3  retrains DeepDenoiser / DPRNN / Restormer1D / ProtoCondTCN with the
            SAME interval and the SAME split.
    Step 4  runs the unified comparison and the figures.

    Every training step is skipped when its best checkpoint already exists, so
    the script is safe to re-run after an interruption (Ctrl+C / power loss).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File v9/run_v9_snr_m15_p10_pipeline.ps1 -Profile smoke
    powershell -ExecutionPolicy Bypass -File v9/run_v9_snr_m15_p10_pipeline.ps1 -Profile full
    powershell -ExecutionPolicy Bypass -File v9/run_v9_snr_m15_p10_pipeline.ps1 -Profile full -SkipTraining
#>
param(
    [ValidateSet("smoke", "quick", "full")]
    [string]$Profile = "full",
    [int]$Seed = 0,
    [double]$SnrMinDb = -15.0,
    [double]$SnrMaxDb = 10.0,
    [string]$EventH5 = "E:/p_wave/data/chunk2.hdf5",
    [string]$EventCsv = "E:/p_wave/data/chunk2.csv",
    [string]$NoiseH5 = "E:/p_wave/data/chunk1.hdf5",
    [string]$NoiseCsv = "E:/p_wave/data/chunk1.csv",
    [string]$RunRoot = "v9/experiments/snr_m15_p10_seed0",
    [string]$BaselineRoot = "v3/baeslines/checkpoints_snr_m15_p10",
    [string]$ProtoCondRoot = "v7/retrain_snr_m15_p10/ProtoCondTCN_seed0",
    [string]$CompareDir = "v9/paper_experiments/model_comparison_stead_snr_m15_p10",
    [switch]$SkipV9Training,
    [switch]$SkipBaselineTraining,
    [switch]$SkipTraining,
    [switch]$CudaSafe,
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
    "smoke" {
        $V9Epochs = 2;  $V9Batch = 2; $V9Train = 200;  $V9Val = 100
        $BlEpochs = 1;  $BlBatch = 4; $BlTrain = 200;  $BlVal = 50
        $RsEpochs = 1;  $RsBatch = 4; $RsTrain = 200;  $RsVal = 50
        $PcEpochs = 2;  $PcBatch = 4; $PcTrain = 200
        $CmpSamples = 24
    }
    "quick" {
        $V9Epochs = 5;   $V9Batch = 4; $V9Train = 2000;  $V9Val = 500
        $BlEpochs = 5;   $BlBatch = 8; $BlTrain = 2000;  $BlVal = 200
        $RsEpochs = 5;   $RsBatch = 8; $RsTrain = 2000;  $RsVal = 200
        $PcEpochs = 5;   $PcBatch = 8; $PcTrain = 2000
        $CmpSamples = 200
    }
    "full" {
        $V9Epochs = 30;  $V9Batch = 4; $V9Train = 18000; $V9Val = 2000
        $BlEpochs = 30;  $BlBatch = 8; $BlTrain = 4000;  $BlVal = 1000
        $RsEpochs = 30;  $RsBatch = 8; $RsTrain = 4000;  $RsVal = 1000
        $PcEpochs = 30;  $PcBatch = 8; $PcTrain = 4000
        $CmpSamples = 2000
    }
}

if ($SkipTraining) { $SkipV9Training = $true; $SkipBaselineTraining = $true }

Write-Host ""
Write-Host ("=" * 80)
Write-Host "  PCD-Net V9 -15...10 dB pipeline | profile=$Profile"
Write-Host ("=" * 80)
Write-Host "  python     : $Python"
Write-Host "  input SNR  : [$SnrMinDb, $SnrMaxDb] dB (uniform in dB, pre_event context)"
Write-Host "  run root   : $RunRoot"
Write-Host ""

# ── Step 0: environment ─────────────────────────────────────────────────────
foreach ($File in @($EventH5, $EventCsv, $NoiseH5, $NoiseCsv)) {
    if (-not (Test-Path -LiteralPath $File)) { throw "Required file not found: $File" }
}
& $Python -c "import pywt, torch; print('[env] pywt', pywt.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "PyWavelets is missing: run `$Python -m pip install PyWavelets" }

# ── Step 1: shared hold-out split (seed 0, 10%) ─────────────────────────────
$ValCsv = "v5/v7_seed$($Seed)_val.csv"
if (-not (Test-Path -LiteralPath $ValCsv)) {
    Write-Host "[split] creating $ValCsv from $EventCsv"
    & $Python -c "from v5.train_v6 import split_dataset; print(split_dataset(r'$EventCsv', 0.1, $Seed, 'v7_seed$Seed'))"
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the shared split" }
} else {
    Write-Host "[split] reusing $ValCsv"
}
$CompareCsv = $ValCsv

# ── Step 2: retrain PCD-Net V9 at the target interval ───────────────────────
$V9Best = Join-Path $RunRoot "best_model_v9.pth"
if ($SkipV9Training -or (Test-Path -LiteralPath $V9Best)) {
    Write-Host "[V9] skip (checkpoint exists: $V9Best)"
} else {
    Write-Host ""
    Write-Host "[V9] training, input SNR [$SnrMinDb, $SnrMaxDb] dB, single pass, pre_event"
    $V9Args = @(
        "-u", "-m", "v9.train_v9",
        "--dataset_type", "stead", "--seed", "$Seed",
        "--epochs", "$V9Epochs", "--batch_size", "$V9Batch",
        "--snr_min_db=$SnrMinDb", "--snr_max_db=$SnrMaxDb",
        "--n_refine", "0", "--context_source", "pre_event",
        "--val_samples", "$V9Val", "--print_every", "200",
        "--num_workers", "0",
        "--event_h5", $EventH5, "--event_csv", $EventCsv,
        "--noise_h5", $NoiseH5, "--noise_csv", $NoiseCsv,
        "--save_dir", $RunRoot
    )
    if ($V9Train -gt 0) { $V9Args += @("--train_samples", "$V9Train") }
    if ($CudaSafe) { $V9Args += "--cuda_safe" }
    & $Python @V9Args
    if ($LASTEXITCODE -ne 0) { throw "PCD-Net V9 training failed" }
}

# ── Step 3: retrain the trainable baselines on the same interval/split ──────
$DdBest = Join-Path $BaselineRoot "deep_denoiser/best_model.pth"
$DpBest = Join-Path $BaselineRoot "dprnn/best_model.pth"
$RsBest = Join-Path $BaselineRoot "restormer1d/best_model.pth"
$PcBest = Join-Path $ProtoCondRoot "best_proto_cond_tcn.pth"

if ($SkipBaselineTraining) {
    Write-Host "[baselines] skip"
} else {
    if ((Test-Path -LiteralPath $DdBest) -and (Test-Path -LiteralPath $DpBest)) {
        Write-Host "[baselines] DeepDenoiser/DPRNN skip (checkpoints exist)"
    } else {
        Write-Host ""
        Write-Host "[baselines] DeepDenoiser + DPRNN, input SNR [$SnrMinDb, $SnrMaxDb] dB"
        & $Python -u -m v3.baeslines.train_baseline --model all `
            --seed "$Seed" --epochs "$BlEpochs" --batch_size "$BlBatch" `
            --num_workers 0 --max_train_samples "$BlTrain" --val_samples "$BlVal" `
            --snr_min_db="$SnrMinDb" --snr_max_db="$SnrMaxDb" `
            --event_h5 $EventH5 --event_csv $EventCsv `
            --noise_h5 $NoiseH5 --noise_csv $NoiseCsv `
            --save_root $BaselineRoot
        if ($LASTEXITCODE -ne 0) { throw "DeepDenoiser/DPRNN training failed" }
    }

    if (Test-Path -LiteralPath $RsBest) {
        Write-Host "[baselines] Restormer1D skip (checkpoint exists)"
    } else {
        Write-Host ""
        Write-Host "[baselines] Restormer1D, input SNR [$SnrMinDb, $SnrMaxDb] dB"
        & $Python -u -m v3.baeslines.train_restormer1d `
            --seed "$Seed" --val_frac 0.1 --num_workers 0 `
            --epochs "$RsEpochs" --batch_size "$RsBatch" `
            --max_train_samples "$RsTrain" --val_samples "$RsVal" `
            --snr_min_db="$SnrMinDb" --snr_max_db="$SnrMaxDb" `
            --event_h5 $EventH5 --event_csv $EventCsv `
            --noise_h5 $NoiseH5 --noise_csv $NoiseCsv `
            --save_dir (Join-Path $BaselineRoot "restormer1d")
        if ($LASTEXITCODE -ne 0) { throw "Restormer1D training failed" }
    }

    if (Test-Path -LiteralPath $PcBest) {
        Write-Host "[baselines] ProtoCondTCN skip (checkpoint exists)"
    } else {
        Write-Host ""
        Write-Host "[baselines] ProtoCondTCN, dB-uniform input SNR [$SnrMinDb, $SnrMaxDb]"
        & $Python -u -m v7.proto_cond_tcn_baseline `
            --seed "$Seed" --epochs "$PcEpochs" --batch_size "$PcBatch" `
            --eval_batch_size "$PcBatch" --num_workers 0 --print_every 100 `
            --snr_sampling db `
            --train_snr_min="$SnrMinDb" --train_snr_max="$SnrMaxDb" `
            --train_samples "$PcTrain" --val_samples 500 --eval_samples "$CmpSamples" `
            --event_h5 $EventH5 --event_csv $EventCsv `
            --noise_h5 $NoiseH5 --noise_csv $NoiseCsv `
            --save_dir $ProtoCondRoot
        if ($LASTEXITCODE -ne 0) { throw "ProtoCondTCN training failed" }
    }
}

# ── Step 4: unified comparison + figures ────────────────────────────────────
Write-Host ""
Write-Host "[compare] unified evaluation on $CompareCsv"
& (Join-Path $PSScriptRoot "run_v9_comparison.ps1") `
    -Profile $Profile -Seed $Seed `
    -SnrMinDb $SnrMinDb -SnrMaxDb $SnrMaxDb `
    -EventH5 $EventH5 -EventCsv $CompareCsv `
    -NoiseH5 $NoiseH5 -NoiseCsv $NoiseCsv `
    -V9Checkpoint $V9Best `
    -ProtoCondCkpt $PcBest `
    -DeepDenoiserCkpt $DdBest `
    -DprnnCkpt $DpBest `
    -RestormerCkpt $RsBest `
    -SaveDir $CompareDir `
    -CudaSafe:$CudaSafe `
    -Python $Python
if ($LASTEXITCODE -ne 0) { throw "Comparison failed" }

Write-Host ""
Write-Host "[pipeline done]"
