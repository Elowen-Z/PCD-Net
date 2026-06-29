# PCD-Net: Prototype-Conditioned Denoising for Microseismic Signals

This repository contains the implementation and experiment scripts for PCD-Net,
a prototype-conditioned adaptive denoising network for microseismic and seismic
waveform recovery.

PCD-Net uses an event-preceding background-noise segment to retrieve sparse
noise prototypes, performs prototype-guided cross-attention, refines the output
through residual feedback, and applies a learned quality score for adaptive
inference.

## Main Features

- Noise-condition encoder for event-preceding background noise.
- Top-M sparse prototype selection from a learnable prototype codebook.
- Prototype-conditioned cross-attention for dynamic denoising.
- Residual-feedback iterative refinement.
- Quality-aware adaptive inference and stopping.
- Transfer-learning experiments on mine microseismic and industrial
  non-natural seismic data.

## Repository Structure

```text
PCD-Net/
  v7/                         # Main PCD-Net implementation and experiments
  v5/                         # Minimal reused V6 backbone/data utilities
  v4/                         # Noise encoder and prototype modules
  v3/                         # STEAD dataset loader and baseline models
  paper_results/
    figures/                  # Selected paper figures
    tables/                   # Selected result tables
  requirements.txt
  README.md
  LICENSE
```

## Environment

The code was developed with Python 3.8+ and PyTorch. Install dependencies with:

```bash
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build matching your GPU and CUDA version
from the official PyTorch website.

## Data Preparation

This repository does not include the original datasets or trained checkpoints.

Expected datasets:

- STEAD source-domain earthquake waveforms.
- Mine microseismic target-domain waveforms.
- Industrial non-natural seismic target-domain waveforms.

The default scripts expect HDF5/CSV waveform metadata paths. Update the dataset
paths in command-line arguments or in the configuration section of the scripts.

Typical waveform setting:

- 3 components: Z, N, E.
- Input waveform length: 6000 samples.
- Background-noise condition length: 400 samples.
- Sampling rate used in experiments: 100 Hz.

## Quick Test

```bash
python -m v7.test_v7
```

## STEAD Training

```bash
python -m v7.train_v7 \
  --dataset_type stead \
  --epochs 30 \
  --batch_size 2 \
  --top_m 4 \
  --save_dir checkpoints_feedback_stead_seed0
```

On Windows/CUDA systems where cuDNN or Flash SDP kernels are unstable, use:

```bash
python -m v7.train_v7 \
  --dataset_type stead \
  --epochs 30 \
  --batch_size 2 \
  --top_m 4 \
  --cuda_safe \
  --no_amp \
  --no_pin_memory
```

## Transfer Fine-Tuning

Example for target-domain fine-tuning:

```bash
python -m v7.transfer_staged_v7 \
  --target mining \
  --source_ckpt checkpoints_feedback_stead_seed0/best_model_v7.pth \
  --save_dir transfer_comparisons/mining/pcd_adaptation \
  --stage2_epochs 5 \
  --stage3_epochs 15 \
  --batch_size 2 \
  --freeze_strategy pcd_adaptation
```

Supported freezing strategies include:

- `all_frozen`
- `noise_encoder`
- `prototype_feedback`
- `pcd_adaptation`
- `full`

## Paper Experiments

Selected scripts for reproducing figures and tables:

```bash
python -m v7.plot_full_model_comparison_v7
python -m v7.plot_snr_group_gain_comparison
python -m v7.plot_mining_transfer_finetune_ablation
python -m v7.plot_nonnat_transfer_finetune_ablation
python -m v7.plot_quality_head_effectiveness
```

Selected result tables and final figures are provided under `paper_results/`.

## Notes

- Large datasets, intermediate candidates, and model checkpoints are excluded
  from the repository.
- If you use private industrial or mine monitoring data, place them outside the
  repository and pass paths through command-line arguments.
- Checkpoints can be released separately through GitHub Releases, Zenodo, or
  another file hosting service.

## Citation

If this code is useful for your research, please cite the corresponding paper
once it is available.
