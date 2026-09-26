# PCD-Net V9

V9 replaces Top-M/Top-1 selection with dense soft routing over every prototype.
The default restoration path is a single pass (`n_refine=0` means one initial
pass and zero additional refinement passes), avoiding repeated residual
accumulation.

```text
background -> context encoder -> cosine similarity -> soft alpha
soft alpha -> routing-prior prototype attention -> restoration decoder
```

The attention logits implement:

```text
softmax(QK^T / sqrt(d_h) + lambda_r * log(alpha + eps))
```

Legacy VQ commitment, VQ diversity, and sparse-selection losses default to
zero. The model learns its prototypes through the restoration objective and
the fully differentiable routing/attention path.

Residual feedback and adaptive early stopping are disabled by default because
they require additional refinement passes. They are enabled automatically only
when an experiment explicitly sets `--n_refine` above zero.

`--no_routing` sets `alpha=1/K`, retaining the same prototype attention while
removing background-dependent routing. This is the controlled Table 2 ablation.

## Quick verification

```powershell
python -m v9.test_v9
```

## Main model and routing ablation

The quick profile limits samples and epochs for a short validation run:

```powershell
powershell -ExecutionPolicy Bypass -File v9/run_v9_experiments.ps1 -Profile quick
```

Use `-Profile full` for the paper run.

## Selective target adaptation

```powershell
python -m v9.transfer_v9 `
  --checkpoint v9/experiments/table1_seed0/best_model_v9.pth `
  --dataset_type mining `
  --event_h5 E:/path/to/target.hdf5 `
  --event_csv E:/path/to/target.csv `
  --noise_h5 E:/path/to/noise.hdf5 `
  --noise_csv E:/path/to/noise.csv `
  --strategy environment `
  --save_dir v9/experiments/table3_environment
```

The `environment` strategy updates only the context encoder, prototype memory,
and router. The waveform encoder, decoder, refinement module, and quality head
remain frozen. Use `--strategy full` for the full fine-tuning baseline. Frozen
performance is obtained by evaluating the source checkpoint directly.

The `decoder_only` strategy updates only the waveform decoder blocks
(`backbone.dec4` through `backbone.dec0`) and `backbone.out_head`. The waveform
encoder, prototype cross-attention, context encoder, prototype memory, router,
refinement module, and quality head remain frozen. The adaptation ablation
runner includes this variant in the same held-out evaluation and merged table.

The `environment_decoder` strategy jointly updates the environment branch
(context encoder, prototype memory, and router) and waveform decoder blocks
(`backbone.dec4` through `backbone.dec0`, plus `backbone.out_head`). The waveform
encoder, prototype cross-attention, refinement module, and quality head remain
frozen. The runner reports this variant as `Env + Decoder` in the merged table.

The default context source is `pre_event`, so the background segment comes from
the same waveform before the P arrival. The legacy chunk1 pairing is available
only through an explicit `--context_source paired_noise` argument.

## Environment

Use the working interpreter; the in-repo `.venv` is stale (its base Python 3.9
was removed) and the old `D:\app\anaconda\...` path no longer exists:

```text
D:\app\python\envs\EarthquakeDetection\python.exe   # Python 3.11, torch 2.6.0+cu124
```

`PyWavelets` is required by the classical baselines
(`v3/baeslines/traditional_denoise.py` imports `pywt` at module level):

```powershell
& $Py -m pip install PyWavelets
```

## Input-SNR controlled training (-15 ... 10 dB)

The evaluation protocol has always drawn the input SNR uniformly in dB over
`[-15, 10]` dB (`v5.compare_models.CONFIG["snr_db_range"]`). V9's
`DEFAULT_CONFIG` now matches that interval for training as well
(`snr_range=(-15.0, 10.0)`, `snr_sampling="db"`). State it explicitly so the run
is self-documenting:

```powershell
$Py = 'D:\app\python\envs\EarthquakeDetection\python.exe'
Set-Location D:\X\denoise\part1

& $Py -u -m v9.train_v9 `
  --dataset_type stead --seed 0 `
  --epochs 30 --batch_size 4 `
  --snr_min_db -15 --snr_max_db 10 `
  --n_refine 0 --context_source pre_event `
  --train_samples 18000 --val_samples 2000 --print_every 200 `
  --event_h5 E:/p_wave/data/chunk2.hdf5 --event_csv E:/p_wave/data/chunk2.csv `
  --noise_h5 E:/p_wave/data/chunk1.hdf5 --noise_csv E:/p_wave/data/chunk1.csv `
  --cuda_safe --num_workers 0 `
  --save_dir v9/experiments/snr_m15_p10_seed0
```

Notes:

* Do **not** pass `--train_samples 0`; `make_loaders` turns it into an empty
  subset. Omit the flag to use the full training split.
* `--train_samples` selects the first N rows of the split CSV, so runs with
  different N are not nested and should not be compared directly.
* Checkpoints under `v9/experiments/table1_*` were trained on
  `snr_range=[0.1, 20.0]` with a 2000-sample budget; retrain before comparing.

## Unified comparison (Table 2 / Table 3)

`v9/compare_v9.py` benchmarks Bandpass, Wavelet, DeepDenoiser, DPRNN,
Restormer1D, ProtoCondTCN and PCD-Net on **one shared hold-out split** with a
fixed input-SNR interval. A single `EvalDataset` instance is reused for every
method, so the noisy waveform, the sampled input SNR and the conditioning
window are identical index by index; the runner verifies this by comparing the
per-method input-SNR vector and aborts if it ever differs.

```powershell
powershell -ExecutionPolicy Bypass -File v9/run_v9_comparison.ps1 -Profile full
```

Direct invocation, including a finer low-SNR binning. The value starts with
`-`, so the `--flag=value` form is required:

```powershell
& $Py -u -m v9.compare_v9 `
  --event_h5 E:/p_wave/data/chunk2.hdf5 --event_csv v5/v7_seed0_val.csv `
  --noise_h5 E:/p_wave/data/chunk1.hdf5 --noise_csv E:/p_wave/data/chunk1.csv `
  --context_source pre_event `
  --snr_min_db -15 --snr_max_db 10 --snr_groups='-15,-10,-5,0,5,10' `
  --max_samples 2000 --batch_size 4 --seed 0 --cuda_safe `
  --ckpt_v9             v9/experiments/snr_m15_p10_seed0/best_model_v9.pth `
  --ckpt_deepdenoiser   v3/baeslines/checkpoints_snr_m15_p10/deep_denoiser/best_model.pth `
  --ckpt_dprnn          v3/baeslines/checkpoints_snr_m15_p10/dprnn/best_model.pth `
  --ckpt_restormer1d    v3/baeslines/checkpoints_snr_m15_p10/restormer1d/best_model.pth `
  --ckpt_proto_cond_tcn v7/retrain_snr_m15_p10/ProtoCondTCN_seed0/best_proto_cond_tcn.pth `
  --save_dir v9/paper_experiments/model_comparison_stead_snr_m15_p10
```

`--resume` reuses `per_model_<Name>.csv` files that already hold the expected
number of rows, which makes an interrupted 2000-sample run cheap to restart.
`--allow_missing` skips methods whose checkpoint is absent instead of aborting.

Outputs (the CSV schema matches `v5/compare_models.py`, so the existing
aggregation and plotting scripts keep working):

```text
per_model_<Name>.csv               per-sample metrics
global_compare.csv                 Table 2
per_group_compare.csv              Table 3 (DeltaSNR by input-SNR group)
paper_model_comparison_table.csv   bootstrap CIs
paper_model_comparison_table.tex   LaTeX
compare_bar.png                    six-panel overview
summary.txt                        text summary
run_config.json                    exact protocol record
test_set_audit.json                per-method N / SNR_in / SNR-group counts
```

Figures:

```powershell
& $Py -m v7.plot_snr_group_gain_comparison `
  --input   v9/paper_experiments/model_comparison_stead_snr_m15_p10/per_group_compare.csv `
  --out_dir v9/paper_experiments/model_comparison_stead_snr_m15_p10/plots
& $Py -m v7.plot_full_model_comparison_v7 `
  --csv     v9/paper_experiments/model_comparison_stead_snr_m15_p10/paper_model_comparison_table.csv `
  --out_dir v9/paper_experiments/model_comparison_stead_snr_m15_p10/plots
```

`plot_snr_group_gain_comparison.py` auto-detects the group order from the input
CSV, so custom `--snr_groups` binning is picked up without editing the script.

## Full pipeline (train everything, then compare)

```powershell
# smoke: 1-2 epochs, 200 training samples, 24 evaluation samples
powershell -ExecutionPolicy Bypass -File v9/run_v9_snr_m15_p10_pipeline.ps1 -Profile smoke

# full: V9 18000 samples x 30 epochs, baselines 4000 x 30, 2000 evaluation samples
powershell -ExecutionPolicy Bypass -File v9/run_v9_snr_m15_p10_pipeline.ps1 -Profile full

# comparison only (all checkpoints already exist)
powershell -ExecutionPolicy Bypass -File v9/run_v9_snr_m15_p10_pipeline.ps1 -Profile full -SkipTraining
```

Every training step is skipped when its best checkpoint already exists, so the
script is safe to re-run after an interruption.

### Baseline training at the same interval

`--snr_min_db / --snr_max_db / --snr_sampling` were added to the baseline
trainers so they can share the V9 protocol. Defaults preserve the historic
behaviour exactly (linear power ratios, uniform in linear power):

```powershell
& $Py -u -m v3.baeslines.train_baseline --model all --seed 0 `
  --epochs 30 --batch_size 8 --max_train_samples 4000 --val_samples 1000 `
  --snr_min_db -15 --snr_max_db 10 `
  --save_root v3/baeslines/checkpoints_snr_m15_p10

& $Py -u -m v3.baeslines.train_restormer1d --seed 0 --val_frac 0.1 `
  --epochs 30 --batch_size 8 --max_train_samples 4000 --val_samples 1000 `
  --snr_min_db -15 --snr_max_db 10 `
  --save_dir v3/baeslines/checkpoints_snr_m15_p10/restormer1d

& $Py -u -m v7.proto_cond_tcn_baseline --seed 0 --epochs 30 --batch_size 8 `
  --snr_sampling db --train_snr_min=-15 --train_snr_max=10 `
  --train_samples 4000 --val_samples 500 --eval_samples 2000 `
  --save_dir v7/retrain_snr_m15_p10/ProtoCondTCN_seed0
```

Warnings:

* Always pass `--seed 0` to the baseline trainers. Their internal split is
  `df.sample(frac=0.1, random_state=seed)`; with the historic seed 42 the
  resulting hold-out is a *different* 10% (only ~1924 of 20000 `trace_name`
  values overlap with the V9 split), which leaks about 90% of the evaluation
  set into the baseline training set. `train_restormer1d.py` additionally needs
  `--val_frac 0.1`.
* `v3/baeslines/train_baseline.py` resumes from
  `--save_root/<model>/best_model.pth` when that file exists, so always point
  `--save_root` at a fresh directory when retraining.
* `--snr_sampling db` is what makes ProtoCondTCN sample uniformly in dB.
  Without it, `--train_snr_is_db` only converts the bounds to power ratios and
  the sampler stays uniform in linear power.
