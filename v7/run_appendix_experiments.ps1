param(
    [ValidateSet("all", "qualitative", "snr_table", "complexity", "refine_threshold", "prototype")]
    [string]$Experiment = "all",
    [string]$Checkpoint = "v7\checkpoints_feedback_stead_seed0\best_model_v7.pth",
    [string]$Domains = "stead,mining,nonnat",
    [string]$NoiseTypes = "mixed",
    [string]$SnrLevels = "-10,-5,0,5,10,15",
    [int]$MaxSamples = 200,
    [int]$BatchSize = 1,
    [int]$ExamplesPerCondition = 3,
    [int]$SensitivitySamples = 500,
    [int]$LatencyRepetitions = 30,
    [switch]$FastCuda,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$Python = "D:\app\anaconda\anaconda\envs\EarthquakeDetection\python.exe"
$Workspace = "D:\X\denoise\part1"

Set-Location $Workspace

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "========== $Name =========="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$CommonCuda = @()
if ($FastCuda) { $CommonCuda += "--fast_cuda" }
if ($Cpu) { $CommonCuda += "--cpu" }

if ($Experiment -eq "all" -or $Experiment -eq "qualitative") {
    Invoke-Step "Appendix qualitative examples" {
        & $Python -u -m v7.appendix_qualitative_examples_v7 `
            "--checkpoint=$Checkpoint" `
            "--domains=$Domains" `
            "--noise_types=$NoiseTypes" `
            "--snr_levels=$SnrLevels" `
            "--examples_per_condition=$ExamplesPerCondition" `
            "--output_dir=v7/paper_experiments/appendix/qualitative_examples" `
            @CommonCuda
    }
}

if ($Experiment -eq "all" -or $Experiment -eq "snr_table") {
    Invoke-Step "Appendix complete SNR table" {
        & $Python -u -m v7.run_medical_paper_style_experiments `
            "--checkpoint=$Checkpoint" `
            "--domains=$Domains" `
            "--noise_types=white,pink,recorded,mixed" `
            "--snr_levels=$SnrLevels" `
            "--max_samples=$MaxSamples" `
            "--batch_size=$BatchSize" `
            "--output_dir=v7/paper_experiments/appendix/snr_complete_table" `
            "--skip_ablation" `
            "--skip_module_ablation" `
            "--skip_complexity" `
            "--restart" `
            $(if ($FastCuda) { "--fast_cuda" })
    }
}

if ($Experiment -eq "all" -or $Experiment -eq "complexity") {
    Invoke-Step "Appendix complexity and latency" {
        & $Python -u -m v7.run_medical_paper_style_experiments `
            "--checkpoint=$Checkpoint" `
            "--domains=stead" `
            "--noise_types=mixed" `
            "--snr_levels=-5" `
            "--max_samples=1" `
            "--batch_size=1" `
            "--latency_repetitions=$LatencyRepetitions" `
            "--output_dir=v7/paper_experiments/appendix/complexity_latency" `
            "--skip_ablation" `
            "--skip_module_ablation" `
            "--restart" `
            $(if ($FastCuda) { "--fast_cuda" })
    }
}

if ($Experiment -eq "all" -or $Experiment -eq "refine_threshold") {
    Invoke-Step "Appendix refinement and threshold sensitivity" {
        & $Python -u -m v7.appendix_sensitivity_v7 `
            "--checkpoint=$Checkpoint" `
            "--dataset_type=stead" `
            "--max_samples=$SensitivitySamples" `
            "--batch_size=$BatchSize" `
            "--experiments=refine,threshold" `
            "--output_dir=v7/paper_experiments/appendix/refine_threshold_sensitivity" `
            @CommonCuda
    }
}

if ($Experiment -eq "all" -or $Experiment -eq "prototype") {
    Invoke-Step "Appendix prototype and Top-M sensitivity" {
        & $Python -u -m v7.appendix_sensitivity_v7 `
            "--checkpoint=$Checkpoint" `
            "--dataset_type=stead" `
            "--max_samples=$SensitivitySamples" `
            "--batch_size=$BatchSize" `
            "--experiments=topm,keff" `
            "--output_dir=v7/paper_experiments/appendix/prototype_topm_sensitivity" `
            @CommonCuda
    }
}

Write-Host ""
Write-Host "[DONE] Appendix experiments finished."
