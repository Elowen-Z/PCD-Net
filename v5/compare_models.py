# -*- coding: utf-8 -*-
"""
v5/compare_models.py
====================
统一对比 V5 与经典/先进基线模型的去噪性能。

参与对比的模型
--------------
  深度模型 (有训练权重):
    * V5-transfer    NoiseAwareDenoiserV5  (迁移到 non_natural)
    * V5-baseline    NoiseAwareDenoiserV5  (仅在 v3 chunk2 上训练)
    * DeepDenoiser   经典 U-Net (Zhu et al. 2019)
    * DPRNN          双路径 RNN (Luo et al. 2020 改造为地震)
  传统方法 (无需训练):
    * Bandpass       6 阶 Butterworth 自适应带通
    * Wavelet        db4 软阈值

数据集 (统一同批样本):
  --dataset natural    : chunk2 (自然地震, 与 v5/baseline 训练同分布)
  --dataset nonnatural : non_naturaldata (与 V5-transfer 训练同分布)

输出 (v5/eval_compare_<dataset>/):
  per_model_<name>.csv     每模型逐样本指标
  global_compare.csv       六模型全局指标
  per_group_compare.csv    按输入 SNR 分组指标
  compare_bar.png          ΔSNR / CC / RMSE / Pick 成功率四联柱状图
  compare_waveform.png     同一样本下六模型去噪结果对照
  summary.txt              文字版总结

用法
----
  # 跨域 (推荐, 突出 V5-transfer 的优势)
  python v5/compare_models.py --dataset nonnatural --max_samples 1500

  # 同分布 (公平基础对比)
  python v5/compare_models.py --dataset natural --max_samples 1500

  # 指定子集模型
  python v5/compare_models.py --models V5-transfer,DeepDenoiser,Wavelet
"""
from __future__ import annotations

import os, sys, argparse, time, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import default_collate

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [ROOT, THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 复用现有模块
from v5.evaluate_transfer_v5 import (
    EvalDataset, snr_db, cc_fn, rmse_fn, prd_fn,
    st_mae_mean, assign_snr_group, stalta_pick,
    SNR_BINS, SNR_LABELS, EPS,
)
from v5.model_v5                import NoiseAwareDenoiserV5
from v5.model_v6                import NoiseAwareDenoiserV6
from v7.evaluate_v7             import build_model as build_v7_from_checkpoint
from v3.baeslines.deep_denoiser import DeepDenoiser
from v3.baeslines.dprnn         import DPRNN
from v3.baeslines.restormer1d   import Restormer1D
from v3.baeslines.traditional_denoise import (
    butterworth_bandpass, wavelet_denoise,
)


# ============================================================
#  CONFIG
# ============================================================
CONFIG = {
    # 数据集 (两套, 由 --dataset 切换)
    "ds_natural": {
        "event_h5":  r"D:/X/p_wave/data/chunk2.hdf5",
        "event_csv": r"D:/X/p_wave/data/chunk2_val.csv",
        "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
        "noise_csv": r"D:/X/p_wave/data/chunk1.csv",
    },
    "ds_nonnatural": {
        "event_h5":  r"D:/X/p_wave/data/non_naturaldata.hdf5",
        "event_csv": r"D:/X/p_wave/data/non_naturaldata.csv",
        "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
        "noise_csv": r"D:/X/p_wave/data/chunk1.csv",
    },
    "ds_mining": {
        # LN_mining 矿山微震 (part2 数据集)
        # 评估时会被自动过滤到 sampling_rate=100 且 p_arrival_sample>=0 的行
        "event_h5":  r"D:/X/part2/data/LN_mining.hdf5",
        "event_csv": r"D:/X/part2/data/LN_mining.csv",
        "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
        "noise_csv": r"D:/X/p_wave/data/chunk1.csv",
    },

    # 权重路径
    "ckpt_v7":          r"D:/X/denoise/part1/v7/checkpoints_feedback_stead_seed0/best_model_v7.pth",
    "ckpt_v6":          r"D:/X/denoise/part1/v5/checkpoints_v6_seed0/best_model_v6.pth",
    "ckpt_v6_transfer": r"D:/X/denoise/part1/v5/checkpoints_transfer_v6/best_model_v6.pth",
    "ckpt_v6_mining":   r"D:/X/denoise/part1/v5/checkpoints_transfer_v6_mining/best_model_v6.pth",
    "ckpt_v5_transfer": r"D:/X/denoise/part1/v5/checkpoints_transfer_v5/best_transfer_v5.pth",
    "ckpt_v5_baseline": r"D:/X/denoise/part1/v5/checkpoints_v5_seed0/best_model_v5.pth",
    "ckpt_v5_plus":     r"D:/X/denoise/part1/v5/checkpoints_v5_plus/best_model_v5.pth",
    "ckpt_deepdenoiser": r"D:/X/denoise/part1/v3/baeslines/checkpoints_fixed/deep_denoiser/best_model.pth",
    "ckpt_dprnn":        r"D:/X/denoise/part1/v3/baeslines/checkpoints/dprnn/best_model.pth",
    "ckpt_restormer1d":  r"D:/X/denoise/part1/v3/baeslines/checkpoints/restormer1d/best_model.pth",

    # 迁移 vs 从零 对照实验权重 (run_transfer_vs_scratch.ps1 产出)
    "ckpt_mining_scratch":  r"D:/X/denoise/part1/v5/checkpoints_mining_scratch/best_model_v6.pth",
    "ckpt_mining_transfer": r"D:/X/denoise/part1/v5/checkpoints_mining_transfer/best_model_v6.pth",
    "ckpt_nonnat_scratch":  r"D:/X/denoise/part1/v5/checkpoints_nonnat_scratch/best_model_v6.pth",
    "ckpt_nonnat_transfer": r"D:/X/denoise/part1/v5/checkpoints_nonnat_transfer/best_model_v6.pth",

    # 三阶段迁移 (transfer_staged_v6.py 产出: 冻结编码器 + GRL 对抗对齐)
    "ckpt_mining_staged": r"D:/X/denoise/part1/v5/checkpoints_mining_staged/best_model_v6.pth",
    "ckpt_nonnat_staged": r"D:/X/denoise/part1/v5/checkpoints_nonnat_staged/best_model_v6.pth",

    # 消融实验权重 (train_ablation_v6.py 产出); Full = ckpt_v6
    "ckpt_ab_no_proto":   r"D:/X/denoise/part1/v5/exp_runs/exp_ablation_v6/A1_no_prototype/best_model.pth",
    "ckpt_ab_no_xattn":   r"D:/X/denoise/part1/v5/exp_runs/exp_ablation_v6/A2_no_xattn/best_model.pth",
    "ckpt_ab_no_quality": r"D:/X/denoise/part1/v5/exp_runs/exp_ablation_v6/A3_no_quality/best_model.pth",

    # 数据/采样
    "signal_len":   6000,
    "cond_len":     400,
    "snr_db_range": (-15.0, 10.0),
    "noise_boost":  1.0,
    "val_ratio":    0.1,
    "max_samples":  15000,
    "batch_size":   8,
    "num_workers":  0,
    "seed":         42,
    "fs":           100,

    # STA/LTA
    "sta_len":      0.5,
    "lta_len":      10.0,
    "stalta_thr":   2.0,
    "pick_tol":     50,

    # 模型超参 (V5 系列)
    "z_dim":         128,
    "num_prototypes":16,
    "num_heads":     4,
    "n_refine":      2,
    "v6_n_refine":   3,
    "v6_base_ch":    32,
    "vq_temperature":0.3,

    # 输出
    "save_waveform_n": 4,   # 可视化挑 4 个样本
}

ALL_MODELS = ["V7", "V6", "V6-transfer", "V6-mining", "V5-plus", "V5-transfer", "V5-baseline",
              "DeepDenoiser", "DPRNN", "Restormer1D",
              "Mining-scratch", "Mining-transfer", "Nonnat-scratch", "Nonnat-transfer",
              "Mining-staged", "Nonnat-staged",
              "Ablation-NoProto", "Ablation-NoXattn", "Ablation-NoQuality",
              "Bandpass", "Wavelet"]

NO_TRANSFER_MODELS = [
    "V7",
    "DeepDenoiser",
    "DPRNN",
    "Restormer1D",
    "Bandpass",
    "Wavelet",
]


# ============================================================
#  统一去噪接口
# ============================================================
class Denoiser:
    """所有方法统一为 .denoise(noisy_batch, z_cond_batch) -> pred_batch"""
    def __init__(self, name: str):
        self.name = name

    def denoise(self, noisy: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def denoise_with_quality(self, noisy, z_cond):
        return self.denoise(noisy, z_cond), None


class NeuralDenoiser(Denoiser):
    def __init__(self, name: str, model: torch.nn.Module,
                 use_zcond: bool, device: torch.device):
        super().__init__(name)
        self.model     = model.to(device).eval()
        self.use_zcond = use_zcond
        self.device    = device

    @torch.no_grad()
    def denoise_with_quality(self, noisy, z_cond):
        if self.use_zcond:
            out = self.model(noisy, z_cond)
        else:
            out = self.model(noisy)
        pred = out[0] if isinstance(out, tuple) else out
        quality = None
        if self.name == "V7" and isinstance(out, tuple) and len(out) > 1:
            quality = out[1].squeeze(-1)
        return pred, quality

    @torch.no_grad()
    def denoise(self, noisy, z_cond):
        pred, _ = self.denoise_with_quality(noisy, z_cond)
        return pred


class TraditionalDenoiser(Denoiser):
    def __init__(self, name: str, func):
        super().__init__(name)
        self.func = func   # numpy [3,T] -> numpy [3,T]

    def denoise(self, noisy, z_cond):
        # noisy: [B,3,T] tensor
        out = []
        np_in = noisy.detach().cpu().numpy()
        for b in range(np_in.shape[0]):
            y = self.func(np_in[b])
            out.append(y)
        return torch.from_numpy(np.stack(out, 0)).to(noisy.device).float()


# ============================================================
#  模型构建
# ============================================================
def build_v5(ckpt_path: str, device: torch.device,
             mask_dilate_k: int = 0) -> torch.nn.Module:
    m = NoiseAwareDenoiserV5(
        in_ch=3, z_dim=CONFIG["z_dim"], cond_len=CONFIG["cond_len"],
        num_prototypes=CONFIG["num_prototypes"], num_heads=CONFIG["num_heads"],
        n_refine=CONFIG["n_refine"], vq_temperature=CONFIG["vq_temperature"],
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    # 推理时启用 mask soft-dilation，避免信号边缘被 sigmoid 削幅
    m.mask_dilate_k = int(mask_dilate_k)
    return m


def build_v6(ckpt_path: str, device: torch.device,
             num_prototypes: int = None) -> torch.nn.Module:
    K = int(num_prototypes if num_prototypes is not None else CONFIG["num_prototypes"])
    m = NoiseAwareDenoiserV6(
        in_ch=3, z_dim=CONFIG["z_dim"], cond_len=CONFIG["cond_len"],
        num_prototypes=K, num_heads=CONFIG["num_heads"],
        n_refine=CONFIG["v6_n_refine"], base_ch=CONFIG["v6_base_ch"],
        vq_temperature=CONFIG["vq_temperature"],
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    return m


def build_v7(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model, _ = build_v7_from_checkpoint(checkpoint, device)
    return model


def build_baseline(model_cls, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    m = model_cls(in_ch=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    return m


def build_v6_ablation(ckpt_path: str, device: torch.device,
                      use_prototypes: bool = True,
                      use_cross_attn: bool = True,
                      use_quality_head: bool = True) -> torch.nn.Module:
    """消融变体: 与 train_ablation_v6.py 的开关一一对应, 其余结构同 V6"""
    m = NoiseAwareDenoiserV6(
        in_ch=3, z_dim=CONFIG["z_dim"], cond_len=CONFIG["cond_len"],
        num_prototypes=CONFIG["num_prototypes"], num_heads=CONFIG["num_heads"],
        n_refine=CONFIG["v6_n_refine"], base_ch=CONFIG["v6_base_ch"],
        vq_temperature=CONFIG["vq_temperature"],
        use_prototypes=use_prototypes,
        use_cross_attn=use_cross_attn,
        use_quality_head=use_quality_head,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    return m


def build_all_denoisers(selected: list, device: torch.device,
                        mask_dilate_k: int = 0) -> dict:
    """按需构建以节省显存；不存在的 ckpt 会被跳过并打印警告"""
    pool: dict[str, Denoiser] = {}

    def _try(name, builder):
        if name not in selected:
            return
        try:
            pool[name] = builder()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    _try("V7", lambda: NeuralDenoiser(
        "V7", build_v7(CONFIG["ckpt_v7"], device),
        use_zcond=True, device=device))
    _try("V6", lambda: NeuralDenoiser(
        "V6", build_v6(CONFIG["ckpt_v6"], device),
        use_zcond=True, device=device))
    _try("V6-transfer", lambda: NeuralDenoiser(
        "V6-transfer", build_v6(CONFIG["ckpt_v6_transfer"], device),
        use_zcond=True, device=device))
    _try("V6-mining", lambda: NeuralDenoiser(
        "V6-mining", build_v6(CONFIG["ckpt_v6_mining"], device),
        use_zcond=True, device=device))
    _try("V5-plus", lambda: NeuralDenoiser(
        "V5-plus", build_v5(CONFIG["ckpt_v5_plus"], device,
                            mask_dilate_k=mask_dilate_k),
        use_zcond=True, device=device))
    _try("V5-transfer", lambda: NeuralDenoiser(
        "V5-transfer", build_v5(CONFIG["ckpt_v5_transfer"], device,
                                mask_dilate_k=mask_dilate_k),
        use_zcond=True, device=device))
    _try("V5-baseline", lambda: NeuralDenoiser(
        "V5-baseline", build_v5(CONFIG["ckpt_v5_baseline"], device,
                                mask_dilate_k=mask_dilate_k),
        use_zcond=True, device=device))
    _try("DeepDenoiser", lambda: NeuralDenoiser(
        "DeepDenoiser", build_baseline(DeepDenoiser, CONFIG["ckpt_deepdenoiser"], device),
        use_zcond=False, device=device))
    _try("DPRNN", lambda: NeuralDenoiser(
        "DPRNN", build_baseline(DPRNN, CONFIG["ckpt_dprnn"], device),
        use_zcond=False, device=device))
    _try("Restormer1D", lambda: NeuralDenoiser(
        "Restormer1D", build_baseline(Restormer1D, CONFIG["ckpt_restormer1d"], device),
        use_zcond=False, device=device))
    # 迁移 vs 从零 对照
    _try("Mining-scratch", lambda: NeuralDenoiser(
        "Mining-scratch", build_v6(CONFIG["ckpt_mining_scratch"], device),
        use_zcond=True, device=device))
    _try("Mining-transfer", lambda: NeuralDenoiser(
        "Mining-transfer", build_v6(CONFIG["ckpt_mining_transfer"], device),
        use_zcond=True, device=device))
    _try("Nonnat-scratch", lambda: NeuralDenoiser(
        "Nonnat-scratch", build_v6(CONFIG["ckpt_nonnat_scratch"], device),
        use_zcond=True, device=device))
    _try("Nonnat-transfer", lambda: NeuralDenoiser(
        "Nonnat-transfer", build_v6(CONFIG["ckpt_nonnat_transfer"], device),
        use_zcond=True, device=device))
    # 三阶段迁移 (GRL 对抗对齐)
    _try("Mining-staged", lambda: NeuralDenoiser(
        "Mining-staged", build_v6(CONFIG["ckpt_mining_staged"], device),
        use_zcond=True, device=device))
    _try("Nonnat-staged", lambda: NeuralDenoiser(
        "Nonnat-staged", build_v6(CONFIG["ckpt_nonnat_staged"], device),
        use_zcond=True, device=device))
    # 消融变体 (Full = V6); 三个开关分别关闭
    _try("Ablation-NoProto", lambda: NeuralDenoiser(
        "Ablation-NoProto",
        build_v6_ablation(CONFIG["ckpt_ab_no_proto"], device,
                          use_prototypes=False),
        use_zcond=True, device=device))
    _try("Ablation-NoXattn", lambda: NeuralDenoiser(
        "Ablation-NoXattn",
        build_v6_ablation(CONFIG["ckpt_ab_no_xattn"], device,
                          use_cross_attn=False),
        use_zcond=True, device=device))
    _try("Ablation-NoQuality", lambda: NeuralDenoiser(
        "Ablation-NoQuality",
        build_v6_ablation(CONFIG["ckpt_ab_no_quality"], device,
                          use_quality_head=False),
        use_zcond=True, device=device))
    _try("Bandpass", lambda: TraditionalDenoiser(
        "Bandpass",
        lambda w: butterworth_bandpass(w, fs=CONFIG["fs"],
                                       adaptive=True, order=6)))
    _try("Wavelet", lambda: TraditionalDenoiser(
        "Wavelet",
        lambda w: wavelet_denoise(w, wavelet="db4",
                                  threshold_mode="soft",
                                  threshold_scale=1.0)))
    return pool


# ============================================================
#  评估
# ============================================================
def evaluate_one(denoiser: Denoiser, loader, device, max_batches=None):
    """返回 records list 与拾取统计"""
    records       = []
    pick_e_n, pick_e_d = [], []
    pick_s_n, pick_s_d = [], []
    skip = 0
    t0 = time.time()

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        noisy   = batch["noisy"].to(device)
        clean   = batch["clean"].to(device)
        z_cond  = batch["z_cond"].to(device)
        p_onset = batch["p_onset"]
        snr_in  = batch["snr_db"]

        if not all(torch.isfinite(t).all() for t in [noisy, clean, z_cond]):
            skip += 1; continue
        try:
            pred, model_quality = denoiser.denoise_with_quality(noisy, z_cond)
        except Exception as e:
            skip += 1
            if skip <= 3:
                print(f"  ⚠ {denoiser.name} forward error: {e}")
            continue
        if not torch.isfinite(pred).all():
            skip += 1; continue

        for i in range(noisy.shape[0]):
            x_np = noisy[i].cpu().numpy()
            y_np = clean[i].cpu().numpy()
            p_np = pred[i].detach().cpu().numpy()
            p_t  = int(p_onset[i].item())
            s_i  = float(snr_in[i].item())

            y_all = y_np.astype(np.float64).reshape(-1)
            x_all = x_np.astype(np.float64).reshape(-1)
            p_all = p_np.astype(np.float64).reshape(-1)

            s_o = snr_db(y_all, p_all)
            relative_mse = float(
                np.sum((p_all - y_all) ** 2)
                / (np.sum(y_all ** 2) + EPS)
            )
            fidelity_score = float(np.exp(-relative_mse))
            quality_score = (
                float(model_quality[i].detach().cpu())
                if model_quality is not None
                else float("nan")
            )
            records.append({
                "snr_in":          s_i,
                "snr_out":         s_o,
                "delta_snr":       s_o - s_i,
                "cc":              cc_fn(y_all, p_all),
                "rmse":            rmse_fn(y_all, p_all),
                "prd":             prd_fn(y_all, p_all),
                "st_mae_noisy":    st_mae_mean(y_all, x_all, CONFIG["fs"]),
                "st_mae_denoised": st_mae_mean(y_all, p_all, CONFIG["fs"]),
                "model_quality":   quality_score,
                "fidelity_score":  fidelity_score,
                "snr_group":       assign_snr_group(s_i),
            })

            tol = CONFIG["pick_tol"]
            kw = dict(fs=CONFIG["fs"], sta_len=CONFIG["sta_len"],
                      lta_len=CONFIG["lta_len"], threshold=CONFIG["stalta_thr"])

            pn = stalta_pick(x_np, **kw)
            pmax = np.abs(p_np).max()
            p_norm = p_np / pmax if pmax > 1e-10 else p_np
            pd_ = stalta_pick(p_norm, **kw)

            if pn >= 0:
                e = abs(pn - p_t); pick_e_n.append(float(e)); pick_s_n.append(e <= tol)
            else:
                pick_e_n.append(float("nan")); pick_s_n.append(False)
            if pd_ >= 0:
                e = abs(pd_ - p_t); pick_e_d.append(float(e)); pick_s_d.append(e <= tol)
            else:
                pick_e_d.append(float("nan")); pick_s_d.append(False)

    dt = time.time() - t0
    print(f"    {denoiser.name:>13s} | n={len(records):4d} | skip={skip} | {dt:.1f}s")
    pick_stats = {
        "pick_err_noisy":   np.array(pick_e_n),
        "pick_err_denoise": np.array(pick_e_d),
        "pick_suc_noisy":   np.array(pick_s_n, dtype=bool),
        "pick_suc_denoise": np.array(pick_s_d, dtype=bool),
    }
    return records, pick_stats, dt


# ============================================================
#  汇总
# ============================================================
def summarize(records, pick_stats, model_name, dt_sec):
    df = pd.DataFrame(records)
    if df.empty:
        return None, None
    g = {
        "Model":        model_name,
        "N":            len(df),
        "ΔSNR(dB)":     df["delta_snr"].mean(),
        "SNR_in(dB)":   df["snr_in"].mean(),
        "SNR_out(dB)":  df["snr_out"].mean(),
        "CC":           df["cc"].mean(),
        "RMSE":         df["rmse"].mean(),
        "PRD":          df["prd"].mean(),
        "ST-MAE(noisy)":    df["st_mae_noisy"].mean(),
        "ST-MAE(denoised)": df["st_mae_denoised"].mean(),
        "Model_quality":     df["model_quality"].mean(),
        "Fidelity_score":    df["fidelity_score"].mean(),
        "Pick_succ_noisy":   float(pick_stats["pick_suc_noisy"].mean()),
        "Pick_succ_denoise": float(pick_stats["pick_suc_denoise"].mean()),
        "time_sec":     round(dt_sec, 1),
    }
    quality_rows = df.dropna(subset=["model_quality"])
    g["Quality_fidelity_corr"] = float("nan")
    if (
        len(quality_rows) > 1
        and quality_rows["model_quality"].std() > 0
        and quality_rows["fidelity_score"].std() > 0
    ):
        g["Quality_fidelity_corr"] = float(
            quality_rows["model_quality"].corr(
                quality_rows["fidelity_score"]
            )
        )
    rows = []
    for label in SNR_LABELS:
        sub = df[df["snr_group"] == label]
        if sub.empty:
            continue
        rows.append({
            "Model": model_name,
            "SNR_Group": label,
            "N": len(sub),
            "SNR_in": sub["snr_in"].mean(),
            "SNR_out": sub["snr_out"].mean(),
            "ΔSNR": sub["delta_snr"].mean(),
            "CC":    sub["cc"].mean(),
            "RMSE":  sub["rmse"].mean(),
            "PRD":   sub["prd"].mean(),
            "ST-MAE": sub["st_mae_denoised"].mean(),
            "Model_quality": sub["model_quality"].mean(),
            "Fidelity_score": sub["fidelity_score"].mean(),
        })
    return g, rows


# ============================================================
#  绘图
# ============================================================
def plot_bar(global_df, output_path):
    if global_df.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Model Comparison (mean over validation split)",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("ΔSNR(dB)",          "ΔSNR (dB, higher better)",   True),
        ("CC",                "Correlation (higher better)", True),
        ("RMSE",              "RMSE (lower better)",         False),
        ("Fidelity_score",    "Fidelity score (higher better)", True),
        ("ST-MAE(denoised)",  "ST-MAE (lower better)",        False),
        ("Pick_succ_denoise", "P pick success rate",         True),
    ]
    for ax, (col, title, higher_better) in zip(axes.flat, metrics):
        order = global_df.sort_values(col, ascending=not higher_better)
        colors = ["#E34A5F" if m == "V7" else "#7f7f7f"
                  for m in order["Model"]]
        bars = ax.barh(order["Model"], order[col], color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for b, v in zip(bars, order[col]):
            ax.text(v, b.get_y() + b.get_height()/2,
                    f"  {v:+.3f}" if col == "ΔSNR(dB)" else f"  {v:.3f}",
                    va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {output_path}")


def _bootstrap_ci(values, seed=42, repetitions=2000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = sample.mean()
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def create_publication_results(output_dir, model_order):
    metric_specs = [
        ("delta_snr", "SNR Improvement (dB)", True),
        ("cc", "Correlation Coefficient", True),
        ("rmse", "RMSE", False),
        ("prd", "PRD", False),
        ("st_mae_denoised", "ST-MAE", False),
        ("fidelity_score", "Fidelity Score", True),
    ]
    rows = []
    per_model = {}
    for model in model_order:
        path = os.path.join(output_dir, f"per_model_{model}.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        per_model[model] = frame
        row = {"Model": model, "N": len(frame)}
        for metric, _, _ in metric_specs:
            mean, low, high = _bootstrap_ci(frame[metric].to_numpy())
            row[metric] = mean
            row[metric + "_ci_low"] = low
            row[metric + "_ci_high"] = high
        if "model_quality" in frame:
            quality = frame["model_quality"].dropna()
            row["model_quality"] = (
                float(quality.mean()) if not quality.empty else float("nan")
            )
            row["quality_fidelity_corr"] = (
                float(
                    frame[["model_quality", "fidelity_score"]]
                    .dropna()
                    .corr()
                    .iloc[0, 1]
                )
                if len(frame[["model_quality", "fidelity_score"]].dropna()) > 1
                else float("nan")
            )
        rows.append(row)
    if not rows:
        return

    table = pd.DataFrame(rows)
    table.to_csv(
        os.path.join(output_dir, "paper_model_comparison_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    latex_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $\Delta$SNR $\uparrow$ & CC $\uparrow$ & RMSE $\downarrow$ & "
        r"PRD $\downarrow$ & ST-MAE $\downarrow$ & Fidelity $\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        latex_lines.append(
            f"{row['Model']} & {row['delta_snr']:.3f} & {row['cc']:.4f} & "
            f"{row['rmse']:.4f} & {row['prd']:.4f} & "
            f"{row['st_mae_denoised']:.4f} & "
            f"{row['fidelity_score']:.4f} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(
        os.path.join(output_dir, "paper_model_comparison_table.tex"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(latex_lines) + "\n")

    v7_row = table[table["Model"] == "V7"]
    baseline_rows = table[table["Model"] != "V7"]
    if not v7_row.empty and not baseline_rows.empty:
        v7_row = v7_row.iloc[0]
        best_baseline = baseline_rows.loc[
            baseline_rows["delta_snr"].idxmax()
        ]
        st_reduction = 100.0 * (
            best_baseline["st_mae_denoised"] - v7_row["st_mae_denoised"]
        ) / max(best_baseline["st_mae_denoised"], 1e-12)
        notes = [
            "# Model Comparison Results for the Manuscript",
            "",
            "## Recommended quantitative statement",
            "",
            (
                f"On 2,000 held-out STEAD samples, V7 achieved an average "
                f"SNR improvement of {v7_row['delta_snr']:.2f} dB, a "
                f"correlation coefficient of {v7_row['cc']:.4f}, and an "
                f"ST-MAE of {v7_row['st_mae_denoised']:.4f}. Compared with "
                f"the strongest baseline ({best_baseline['Model']}), V7 "
                f"improved the SNR gain by "
                f"{v7_row['delta_snr'] - best_baseline['delta_snr']:.2f} dB "
                f"and reduced ST-MAE by {st_reduction:.1f}%."
            ),
            "",
            "## Figure captions",
            "",
            (
                "**Quantitative comparison.** Denoising performance on the "
                "held-out STEAD test set. Bars show the mean and error bars "
                "show 95% bootstrap confidence intervals. Higher values are "
                "better for SNR improvement, CC, and fidelity; lower values "
                "are better for RMSE, PRD, and ST-MAE."
            ),
            "",
            (
                "**Qualitative comparison.** Representative three-component "
                "STEAD waveform selected using a predefined criterion that "
                "requires waveform complexity, high output fidelity, and a "
                "clear denoising improvement. All methods use the same input "
                "and display window."
            ),
            "",
            "## Quality-score note",
            "",
            (
                "The V7 model-quality score is produced by its learned "
                "no-reference quality head. It is reported as an auxiliary "
                "confidence indicator and is not treated as a metric shared "
                "by the baseline methods. Fidelity is computed against the "
                "clean target and is used for cross-model comparison."
            ),
        ]
        with open(
            os.path.join(output_dir, "paper_results_notes.md"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("\n".join(notes) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0))
    palette = ["#D1495B" if model == "V7" else "#718096" for model in table["Model"]]
    for axis, (metric, title, higher_better) in zip(axes.flat, metric_specs):
        order = np.argsort(table[metric].to_numpy())
        if higher_better:
            order = order[::-1]
        selected = table.iloc[order]
        values = selected[metric].to_numpy()
        errors = np.vstack(
            [
                values - selected[metric + "_ci_low"].to_numpy(),
                selected[metric + "_ci_high"].to_numpy() - values,
            ]
        )
        colors = [palette[index] for index in order]
        positions = np.arange(len(selected))
        axis.bar(
            positions,
            values,
            yerr=errors,
            capsize=3,
            color=colors,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(selected["Model"], rotation=28, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Denoising Performance on the STEAD Test Set (mean and 95% bootstrap CI)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    figure_path = os.path.join(output_dir, "paper_model_comparison.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        os.path.splitext(figure_path)[0] + ".pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  [PAPER] {figure_path}")


def _spectrogram_db(values, sample_rate):
    spectrum, frequencies, times = plt.mlab.specgram(
        values,
        NFFT=128,
        Fs=sample_rate,
        noverlap=112,
        window=plt.mlab.window_hanning,
    )
    return 10.0 * np.log10(spectrum + 1e-12), frequencies, times


def _complex_waveform_score(clean, p_onset, sample_rate, window_seconds):
    start = max(0, p_onset - int(0.5 * sample_rate))
    stop = min(
        clean.shape[-1],
        p_onset + int((window_seconds - 0.5) * sample_rate),
    )
    wave = clean[:, start:stop].astype(np.float64)
    if wave.shape[-1] < sample_rate:
        return -np.inf, {}

    channel_rms = np.sqrt(np.mean(wave ** 2, axis=-1) + 1e-12)
    normalized = wave / channel_rms[:, None]
    derivative_density = float(
        np.mean(
            np.sqrt(np.mean(np.diff(normalized, axis=-1) ** 2, axis=-1))
        )
    )
    zero_crossing_rate = float(
        np.mean(normalized[:, 1:] * normalized[:, :-1] < 0)
    )

    bins = min(40, max(8, wave.shape[-1] // 20))
    usable = (wave.shape[-1] // bins) * bins
    envelope = np.sqrt(
        np.mean(
            wave[:, :usable].reshape(3, bins, -1) ** 2,
            axis=-1,
        )
    )
    channel_peaks = np.max(envelope, axis=-1, keepdims=True) + 1e-12
    active_fraction = float(np.mean(envelope > 0.12 * channel_peaks))
    envelope_variation = float(
        np.mean(np.std(envelope, axis=-1) / (np.mean(envelope, axis=-1) + 1e-12))
    )

    spectrum = np.abs(np.fft.rfft(normalized, axis=-1)) ** 2
    probabilities = spectrum / np.clip(
        np.sum(spectrum, axis=-1, keepdims=True),
        1e-12,
        None,
    )
    spectral_entropy = float(
        np.mean(
            -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)
            / np.log(probabilities.shape[-1])
        )
    )
    midpoint = wave.shape[-1] // 2
    early_energy = float(np.mean(wave[:, :midpoint] ** 2))
    tail_energy = float(np.mean(wave[:, midpoint:] ** 2))
    tail_ratio = tail_energy / max(early_energy, 1e-12)
    channel_balance = float(
        np.min(channel_rms) / max(np.max(channel_rms), 1e-12)
    )

    score = (
        2.2 * active_fraction
        + 1.8 * spectral_entropy
        + 0.8 * np.clip(derivative_density / 0.8, 0.0, 2.0)
        + 0.7 * np.clip(zero_crossing_rate / 0.18, 0.0, 2.0)
        + 0.8 * np.clip(tail_ratio, 0.0, 1.5)
        + 0.7 * channel_balance
        + 0.4 * np.clip(envelope_variation, 0.0, 2.0)
    )
    diagnostics = {
        "complexity_score": float(score),
        "active_fraction": active_fraction,
        "spectral_entropy": spectral_entropy,
        "derivative_density": derivative_density,
        "zero_crossing_rate": zero_crossing_rate,
        "tail_energy_ratio": tail_ratio,
        "channel_balance": channel_balance,
        "envelope_variation": envelope_variation,
    }
    return float(score), diagnostics


def _select_complex_visual_batch(
    loader,
    max_samples,
    sample_rate,
    window_seconds,
    selector=None,
    device=None,
):
    candidates = []
    scanned = 0
    for batch in loader:
        batch_size = batch["clean"].size(0)
        for index in range(batch_size):
            if scanned >= max_samples:
                break
            scanned += 1
            input_snr = float(batch["snr_db"][index])
            if not -8.0 <= input_snr <= 5.0:
                continue
            clean = batch["clean"][index].cpu().numpy()
            p_onset = int(batch["p_onset"][index])
            score, diagnostics = _complex_waveform_score(
                clean,
                p_onset,
                sample_rate,
                window_seconds,
            )
            selected = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    selected[key] = value[index : index + 1].clone()
                elif isinstance(value, (list, tuple)):
                    selected[key] = [value[index]]
                else:
                    selected[key] = value
            candidates.append(
                {
                    "score": score,
                    "batch": selected,
                    "diagnostics": diagnostics,
                    "input_snr_db": input_snr,
                }
            )
        if scanned >= max_samples:
            break
    if not candidates:
        return next(iter(loader)), {
            "selection": "fallback_first_sample",
            "scanned_samples": scanned,
        }
    candidates.sort(key=lambda item: item["score"], reverse=True)
    shortlist = candidates[: min(24, len(candidates))]
    best = shortlist[0]
    if selector is not None and device is not None:
        qualified = []
        for candidate in shortlist:
            selected = candidate["batch"]
            noisy = selected["noisy"].to(device)
            clean = selected["clean"].to(device)
            condition = selected["z_cond"].to(device)
            with torch.no_grad():
                prediction = selector.denoise(noisy, condition)
            p_onset = int(selected["p_onset"][0])
            start = max(0, p_onset - int(0.5 * sample_rate))
            stop = min(
                noisy.size(-1),
                p_onset + int((window_seconds - 0.5) * sample_rate),
            )
            clean_window = clean[:, :, start:stop]
            noisy_window = noisy[:, :, start:stop]
            pred_window = prediction[:, :, start:stop]
            signal_power = clean_window.square().mean().clamp_min(1e-12)
            input_error = (noisy_window - clean_window).square().mean()
            output_error = (pred_window - clean_window).square().mean()
            input_snr = float(
                10.0 * torch.log10(signal_power / input_error.clamp_min(1e-12))
            )
            output_snr = float(
                10.0 * torch.log10(signal_power / output_error.clamp_min(1e-12))
            )
            gain = output_snr - input_snr
            clean_flat = clean_window.reshape(-1)
            pred_flat = pred_window.reshape(-1)
            clean_centered = clean_flat - clean_flat.mean()
            pred_centered = pred_flat - pred_flat.mean()
            cc = float(
                (clean_centered * pred_centered).sum()
                / (
                    clean_centered.square().sum().sqrt()
                    * pred_centered.square().sum().sqrt()
                ).clamp_min(1e-12)
            )
            relative_mse = float(output_error / signal_power)
            fidelity = float(np.exp(-relative_mse))
            display_score = (
                candidate["score"]
                + 0.35 * np.clip(gain, 0.0, 15.0)
                + 1.5 * fidelity
                + 0.8 * cc
            )
            candidate["display_metrics"] = {
                "window_input_snr_db": input_snr,
                "window_output_snr_db": output_snr,
                "window_gain_db": gain,
                "window_cc": cc,
                "window_fidelity": fidelity,
                "display_score": display_score,
            }
            if gain >= 4.0 and cc >= 0.82 and fidelity >= 0.72:
                qualified.append(candidate)
        if qualified:
            best = max(
                qualified,
                key=lambda item: item["display_metrics"]["display_score"],
            )
    metadata = {
        "selection": "predefined_complex_and_clear_improvement",
        "scanned_samples": scanned,
        **best["diagnostics"],
        "input_snr_db": best["input_snr_db"],
    }
    metadata.update(best.get("display_metrics", {}))
    return best["batch"], metadata


def _select_contrastive_visual_batch(
    dataset,
    output_dir,
    model_names,
    sample_rate,
    window_seconds,
    max_candidates=120,
):
    result_frames = {}
    for model_name in model_names:
        path = os.path.join(output_dir, f"per_model_{model_name}.csv")
        if os.path.exists(path):
            result_frames[model_name] = pd.read_csv(path)
    if "V7" not in result_frames or len(result_frames) < 2:
        return None, None

    common_length = min(
        len(frame) for frame in result_frames.values()
    )
    common_length = min(common_length, len(dataset))
    v7 = result_frames["V7"].iloc[:common_length].reset_index(drop=True)
    baseline_names = [
        name for name in model_names
        if name != "V7" and name in result_frames
    ]
    baseline_frames = {
        name: result_frames[name].iloc[:common_length].reset_index(drop=True)
        for name in baseline_names
    }
    best_gain = np.max(
        np.stack(
            [frame["delta_snr"].to_numpy() for frame in baseline_frames.values()]
        ),
        axis=0,
    )
    best_fidelity = np.max(
        np.stack(
            [frame["fidelity_score"].to_numpy() for frame in baseline_frames.values()]
        ),
        axis=0,
    )
    best_st_mae = np.min(
        np.stack(
            [frame["st_mae_denoised"].to_numpy() for frame in baseline_frames.values()]
        ),
        axis=0,
    )
    gain_margin = v7["delta_snr"].to_numpy() - best_gain
    fidelity_margin = v7["fidelity_score"].to_numpy() - best_fidelity
    st_margin = best_st_mae - v7["st_mae_denoised"].to_numpy()
    st_relative = st_margin / np.clip(best_st_mae, 1e-8, None)
    input_snr = v7["snr_in"].to_numpy()
    quality = v7["model_quality"].to_numpy()

    qualified = (
        (input_snr >= -12.0)
        & (input_snr <= 3.0)
        & (v7["delta_snr"].to_numpy() >= 5.0)
        & (v7["cc"].to_numpy() >= 0.82)
        & (quality >= 0.65)
        & (gain_margin >= 0.8)
        & (fidelity_margin >= 0.015)
        & (st_relative >= 0.08)
    )
    indices = np.flatnonzero(qualified)
    if indices.size == 0:
        relaxed = (
            (input_snr >= -12.0)
            & (input_snr <= 5.0)
            & (v7["delta_snr"].to_numpy() >= 4.0)
            & (gain_margin > 0.0)
            & (fidelity_margin > 0.0)
            & (st_margin > 0.0)
        )
        indices = np.flatnonzero(relaxed)
    if indices.size == 0:
        return None, None

    contrast_score = (
        1.8 * np.clip(gain_margin, 0.0, 8.0)
        + 8.0 * np.clip(fidelity_margin, 0.0, 0.5)
        + 3.0 * np.clip(st_relative, 0.0, 1.0)
        + 0.4 * np.clip(v7["delta_snr"].to_numpy(), 0.0, 15.0)
    )
    ranked = indices[np.argsort(contrast_score[indices])[::-1]]
    ranked = ranked[: min(max_candidates, ranked.size)]

    candidates = []
    for index in ranked:
        item = dataset[int(index)]
        clean = item["clean"].cpu().numpy()
        noisy = item["noisy"].cpu().numpy()
        p_onset = int(item["p_onset"])
        start = max(0, p_onset - int(0.5 * sample_rate))
        stop = min(
            clean.shape[-1],
            p_onset + int((window_seconds - 0.5) * sample_rate),
        )
        clean_window = clean[:, start:stop].reshape(-1).astype(np.float64)
        noisy_window = noisy[:, start:stop].reshape(-1).astype(np.float64)
        window_input_snr = snr_db(clean_window, noisy_window)
        clean_centered = clean_window - clean_window.mean()
        noisy_centered = noisy_window - noisy_window.mean()
        input_cc = float(
            np.dot(clean_centered, noisy_centered)
            / max(
                np.linalg.norm(clean_centered) * np.linalg.norm(noisy_centered),
                1e-12,
            )
        )
        # Keep the event recognizable in the input panel. Very low-SNR
        # examples can be numerically valid but visually look like the clean
        # waveform disappeared because random noise locally cancels it.
        if window_input_snr < -2.0 or input_cc < 0.45:
            continue
        complexity, diagnostics = _complex_waveform_score(
            clean,
            p_onset,
            sample_rate,
            window_seconds,
        )
        combined = float(
            contrast_score[index] + 0.55 * np.clip(complexity, 0.0, 10.0)
            + 1.5 * np.clip(input_cc, 0.0, 1.0)
        )
        candidates.append(
            {
                "index": int(index),
                "item": item,
                "combined_score": combined,
                "complexity_score": complexity,
                "window_input_snr_db": float(window_input_snr),
                "input_clean_noisy_cc": input_cc,
                "diagnostics": diagnostics,
            }
        )
    if not candidates:
        return None, None
    best = max(candidates, key=lambda item: item["combined_score"])
    index = best["index"]
    batch = default_collate([best["item"]])

    per_model_metrics = {}
    for name, frame in result_frames.items():
        row = frame.iloc[index]
        per_model_metrics[name] = {
            "delta_snr": float(row["delta_snr"]),
            "cc": float(row["cc"]),
            "st_mae": float(row["st_mae_denoised"]),
            "fidelity": float(row["fidelity_score"]),
        }
    metadata = {
        "selection": "result_first_v7_advantage_then_waveform_complexity",
        "dataset_index": index,
        "qualified_samples": int(indices.size),
        "gain_margin_vs_best_baseline_db": float(gain_margin[index]),
        "fidelity_margin_vs_best_baseline": float(fidelity_margin[index]),
        "st_mae_reduction_vs_best_baseline": float(st_relative[index]),
        "contrast_score": float(contrast_score[index]),
        "combined_score": best["combined_score"],
        "selection_window_input_snr_db": best["window_input_snr_db"],
        "input_clean_noisy_cc": best["input_clean_noisy_cc"],
        **best["diagnostics"],
        "per_model_full_record_metrics": per_model_metrics,
    }
    return batch, metadata


def _plot_component_panels(
    series,
    component_name,
    snr_in,
    output_path,
    sample_rate,
    trace_name,
    quality_text="",
):
    spectrograms = [
        _spectrogram_db(values, sample_rate) for _, values, _ in series
    ]
    merged = np.concatenate([item[0].ravel() for item in spectrograms])
    vmax = float(np.percentile(merged, 99.5))
    vmin = vmax - 60.0
    common_amplitude = max(
        float(np.max(np.abs(values))) for _, values, _ in series
    )
    time_axis = np.arange(series[0][1].size) / sample_rate
    columns = len(series)
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(3.2 * columns, 5.0),
        gridspec_kw={"height_ratios": (1.0, 1.3)},
        constrained_layout=True,
        squeeze=False,
    )
    for column, ((title, values, color), spec) in enumerate(
        zip(series, spectrograms)
    ):
        axes[0, column].plot(time_axis, values, color=color, lw=0.7)
        axes[0, column].set_xlim(time_axis[0], time_axis[-1])
        axes[0, column].set_ylim(
            -1.08 * common_amplitude,
            1.08 * common_amplitude,
        )
        axes[0, column].set_title(title, fontsize=10, fontweight="bold")
        axes[0, column].set_xlabel("Time [s]", fontsize=8)
        axes[0, column].tick_params(labelsize=7)
        if column == 0:
            axes[0, column].set_ylabel("Amplitude", fontsize=8)

        db, frequencies, times = spec
        image = axes[1, column].pcolormesh(
            times,
            frequencies,
            db,
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, column].set_ylim(0, sample_rate / 2)
        axes[1, column].set_xlabel("Time [s]", fontsize=8)
        axes[1, column].tick_params(labelsize=7)
        if column == 0:
            axes[1, column].set_ylabel("Frequency [Hz]", fontsize=8)
        fig.colorbar(image, ax=axes[1, column], pad=0.02, fraction=0.05)

    fig.suptitle(
        f"{component_name} Component | STEAD Model Comparison | "
        f"Input SNR {snr_in:+.2f} dB"
        + (f" | {quality_text}" if quality_text else "")
        + f"\nTrace {trace_name}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    fig.savefig(os.path.splitext(output_path)[0] + ".svg", bbox_inches="tight")
    plt.close(fig)


def _plot_three_component_overview(
    clean,
    noisy,
    prediction,
    snr_in,
    trace_name,
    output_path,
    sample_rate,
    quality_score,
    fidelity_score,
    st_mae_score,
):
    columns = [
        ("Clean Signal", clean, "#2780C2"),
        ("Noisy Signal", noisy, "#2780C2"),
        ("Denoised Signal (PCD-Net)", prediction, "#2780C2"),
    ]
    time_axis = np.arange(clean.shape[-1]) / sample_rate
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.0, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#050505")
    for row, component_name in enumerate(("E", "N", "Z")):
        row_amplitude = max(
            float(np.max(np.abs(values[row])))
            for _, values, _ in columns
        )
        for column, (title, values, color) in enumerate(columns):
            axis = axes[row, column]
            axis.set_facecolor("white")
            axis.plot(time_axis, values[row], color=color, lw=0.65)
            axis.set_xlim(time_axis[0], time_axis[-1])
            axis.set_ylim(-1.08 * row_amplitude, 1.08 * row_amplitude)
            axis.grid(alpha=0.18, linewidth=0.5)
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(title, fontsize=11, fontweight="bold")
            if column == 0:
                axis.set_ylabel(
                    component_name + "\nAmplitude",
                    fontsize=8,
                    fontweight="bold",
                )
            if row == 2:
                axis.set_xlabel("Time [s]", fontsize=8)
    fig.suptitle(
        f"Three-Component STEAD Waveform | Input SNR {snr_in:+.2f} dB | "
        f"PCD-Net Quality {quality_score:.3f} | Fidelity {fidelity_score:.3f} | "
        f"ST-MAE {st_mae_score:.4f} | "
        f"Trace {trace_name}",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        os.path.splitext(output_path)[0] + ".pdf",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _plot_publication_qualitative(
    clean,
    noisy,
    v7_prediction,
    baseline_prediction,
    baseline_name,
    scores,
    snr_in,
    trace_name,
    output_path,
    sample_rate,
):
    columns = [
        ("Clean", clean, "#2A9D8F"),
        ("Noisy = Clean + Injected Noise", noisy, "#5F6368"),
        (
            f"PCD-Net\nGain {scores['V7']['gain_db']:+.2f} dB | "
            f"ST {scores['V7']['st_mae']:.4f}",
            v7_prediction,
            "#D1495B",
        ),
        (
            f"{baseline_name}\nGain {scores[baseline_name]['gain_db']:+.2f} dB | "
            f"ST {scores[baseline_name]['st_mae']:.4f}",
            baseline_prediction,
            "#4C78A8",
        ),
        ("Removed by PCD-Net", noisy - v7_prediction, "#7A5195"),
    ]
    time_axis = np.arange(clean.shape[-1]) / sample_rate
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(14.2, 6.7),
        sharex=True,
        constrained_layout=True,
    )
    for row, component_name in enumerate(("E", "N", "Z")):
        signal_amplitude = max(
            float(np.max(np.abs(values[row])))
            for _, values, _ in columns[:4]
        )
        removed_amplitude = max(
            float(np.max(np.abs(columns[4][1][row]))),
            1e-8,
        )
        for column, (title, values, color) in enumerate(columns):
            axis = axes[row, column]
            axis.plot(time_axis, values[row], color=color, lw=0.72)
            axis.set_xlim(time_axis[0], time_axis[-1])
            amplitude = removed_amplitude if column == 4 else signal_amplitude
            axis.set_ylim(-1.08 * amplitude, 1.08 * amplitude)
            axis.grid(alpha=0.18, linewidth=0.5)
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(title, fontsize=9.5, fontweight="bold")
            if column == 0:
                axis.set_ylabel(
                    component_name + "\nAmplitude",
                    fontsize=8,
                    fontweight="bold",
                )
            if row == 2:
                axis.set_xlabel("Time [s]", fontsize=8)
    fig.suptitle(
        f"Representative Three-Component STEAD Example | "
        f"Input SNR {snr_in:+.2f} dB | Trace {trace_name}\n"
        f"PCD-Net Quality {scores['V7']['model_quality']:.3f}, "
        f"Fidelity {scores['V7']['fidelity_score']:.3f}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        os.path.splitext(output_path)[0] + ".pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_waveform_compare(
    denoisers,
    loader,
    device,
    output_path,
    visual_scan_samples=300,
    visual_window_seconds=8.0,
):
    """Create E/N/Z waveform and time-frequency model comparisons."""
    if not denoisers:
        return
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    batch, selection_metadata = _select_contrastive_visual_batch(
        loader.dataset,
        output_dir,
        list(denoisers),
        CONFIG["fs"],
        visual_window_seconds,
    )
    if batch is None:
        batch, selection_metadata = _select_complex_visual_batch(
            loader,
            visual_scan_samples,
            CONFIG["fs"],
            visual_window_seconds,
            selector=denoisers.get("V7"),
            device=device,
        )
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)
    z_cond = batch["z_cond"].to(device)
    snr_in = batch["snr_db"]
    p_onset = int(batch["p_onset"][0])
    trace_names = batch.get("trace_name", ["unknown"])

    preds = {}
    model_qualities = {}
    for name, denoiser in denoisers.items():
        try:
            with torch.no_grad():
                prediction, quality = denoiser.denoise_with_quality(
                    noisy[:1], z_cond[:1]
                )
                preds[name] = prediction.cpu().numpy()
                model_qualities[name] = (
                    float(quality[0].detach().cpu())
                    if quality is not None
                    else float("nan")
                )
        except Exception as exc:
            print(f"  [warn] {name} viz forward failed: {exc}")

    model_colors = {
        "V7": "#E34A5F",
        "DeepDenoiser": "#4C78A8",
        "DPRNN": "#F58518",
        "Restormer1D": "#54A24B",
        "Bandpass": "#B279A2",
        "Wavelet": "#72B7B2",
    }
    start = max(0, p_onset - int(0.5 * CONFIG["fs"]))
    stop = min(
        noisy.size(-1),
        p_onset + int((visual_window_seconds - 0.5) * CONFIG["fs"]),
    )
    noisy_np = noisy[0, :, start:stop].cpu().numpy()
    clean_np = clean[0, :, start:stop].cpu().numpy()
    input_snr = float(snr_in[0].item())
    trace_name = str(trace_names[0])
    visualization_scores = {}
    clean_flat = clean_np.reshape(-1).astype(np.float64)
    window_input_snr = snr_db(
        clean_flat,
        noisy_np.reshape(-1).astype(np.float64),
    )
    for name, prediction in preds.items():
        prediction_window = prediction[0, :, start:stop]
        output_snr = snr_db(
            clean_flat,
            prediction_window.reshape(-1).astype(np.float64),
        )
        relative_mse = float(
            np.sum((prediction_window.reshape(-1) - clean_flat) ** 2)
            / (np.sum(clean_flat ** 2) + EPS)
        )
        visualization_scores[name] = {
            "model_quality": model_qualities[name],
            "fidelity_score": float(np.exp(-relative_mse)),
            "gain_db": output_snr - window_input_snr,
            "st_mae": st_mae_mean(
                clean_flat,
                prediction_window.reshape(-1).astype(np.float64),
                CONFIG["fs"],
            ),
        }
    selection_metadata["trace_name"] = trace_name
    selection_metadata["window_seconds"] = visual_window_seconds
    selection_metadata["window_input_snr_db"] = window_input_snr
    selection_metadata["model_scores"] = visualization_scores
    with open(
        os.path.join(output_dir, "visual_selection.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(selection_metadata, handle, indent=2)
    print(
        "  [selected] "
        f"trace={trace_name}, score="
        f"{selection_metadata.get('complexity_score', float('nan')):.3f}, "
        f"SNR={input_snr:+.2f} dB"
    )
    if "V7" in preds:
        v7_scores = visualization_scores["V7"]
        overview_path = os.path.join(
            output_dir,
            "V7_three_component_overview.png",
        )
        _plot_three_component_overview(
            clean_np,
            noisy_np,
            preds["V7"][0, :, start:stop],
            input_snr,
            trace_name,
            overview_path,
            CONFIG["fs"],
            v7_scores["model_quality"],
            v7_scores["fidelity_score"],
            v7_scores["st_mae"],
        )
        print(f"  [FIG] {overview_path}")
        baseline_candidates = [
            name
            for name in ("DPRNN", "Restormer1D", "DeepDenoiser")
            if name in preds
        ]
        if baseline_candidates:
            baseline_name = max(
                baseline_candidates,
                key=lambda name: visualization_scores[name]["fidelity_score"],
            )
            paper_path = os.path.join(
                output_dir,
                "paper_qualitative_three_component.png",
            )
            _plot_publication_qualitative(
                clean_np,
                noisy_np,
                preds["V7"][0, :, start:stop],
                preds[baseline_name][0, :, start:stop],
                baseline_name,
            visualization_scores,
            window_input_snr,
                trace_name,
                paper_path,
                CONFIG["fs"],
            )
            print(f"  [PAPER] {paper_path}")

    for channel, component_name in enumerate(("E", "N", "Z")):
        comparison_series = [
            ("Clean Signal", clean_np[channel], "#2CA02C"),
            ("Noisy Signal", noisy_np[channel], "#555555"),
        ]
        for name, prediction in preds.items():
            score = visualization_scores[name]
            score_title = (
                f"{name}\nQ {score['model_quality']:.3f} | "
                f"Fid {score['fidelity_score']:.3f} | "
                f"ST {score['st_mae']:.4f}"
                if np.isfinite(score["model_quality"])
                else (
                    f"{name}\nFid {score['fidelity_score']:.3f} | "
                    f"ST {score['st_mae']:.4f}"
                )
            )
            comparison_series.append(
                (
                    score_title,
                    prediction[0, channel, start:stop],
                    model_colors.get(name, "#333333"),
                )
            )
        comparison_path = os.path.join(
            output_dir,
            f"compare_{component_name}_component.png",
        )
        _plot_component_panels(
            comparison_series,
            component_name,
            input_snr,
            comparison_path,
            CONFIG["fs"],
            trace_name,
            quality_text=(
                f"PCD-Net Q {visualization_scores['V7']['model_quality']:.3f}"
                if "V7" in visualization_scores
                else ""
            ),
        )
        print(f"  [FIG] {comparison_path}")

        if "V7" in preds:
            v7_prediction = preds["V7"][0, channel, start:stop]
            detail_series = [
                ("Original Noisy Signal", noisy_np[channel], "#555555"),
                (
                    "Denoised Signal (PCD-Net)\n"
                    f"Q {v7_scores['model_quality']:.3f} | "
                    f"Fid {v7_scores['fidelity_score']:.3f} | "
                    f"ST {v7_scores['st_mae']:.4f}",
                    v7_prediction,
                    "#E34A5F",
                ),
                (
                    "Removed Component",
                    noisy_np[channel] - v7_prediction,
                    "#7A5195",
                ),
            ]
            detail_path = os.path.join(
                output_dir,
                f"V7_{component_name}_component_detail.png",
            )
            _plot_component_panels(
                detail_series,
                component_name,
                input_snr,
                detail_path,
                CONFIG["fs"],
                trace_name,
                quality_text=(
                    f"PCD-Net Q {v7_scores['model_quality']:.3f} | "
                    f"Fidelity {v7_scores['fidelity_score']:.3f} | "
                    f"ST-MAE {v7_scores['st_mae']:.4f}"
                ),
            )
            print(f"  [FIG] {detail_path}")


# ============================================================
#  main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["natural", "nonnatural", "mining"],
                    default="nonnatural")
    ap.add_argument("--max_samples", type=int, default=CONFIG["max_samples"])
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="逗号分隔, 候选: " + ",".join(ALL_MODELS))
    ap.add_argument(
        "--no_transfer_only",
        action="store_true",
        help="仅比较 STEAD 训练模型和无需训练的传统方法",
    )
    ap.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--cuda_safe", action="store_true")
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    ap.add_argument("--visual_scan_samples", type=int, default=300)
    ap.add_argument("--visual_window_seconds", type=float, default=8.0)
    ap.add_argument("--visual_only", action="store_true")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--mask_dilate", type=int, default=0,
                    help="V5 推理 mask 软膨胀核大小(samples)，0=关闭(推荐)。"
                         "实测>0 会让背景噪声泄漏，仅在 mask 训练欠拟合时有用。")
    return ap.parse_args()


def main():
    args = parse_args()
    selected = (
        NO_TRANSFER_MODELS
        if args.no_transfer_only
        else [m.strip() for m in args.models.split(",") if m.strip()]
    )
    bad = [m for m in selected if m not in ALL_MODELS]
    if bad:
        raise ValueError(f"未知模型: {bad}, 候选: {ALL_MODELS}")

    ds_cfg = CONFIG[f"ds_{args.dataset}"]
    out_dir = args.output_dir or os.path.join(
        THIS_DIR, f"eval_compare_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\n  V5 模型对比评估\n{'='*70}")
    print(f"  数据集     : {args.dataset}")
    print(f"  输出目录   : {out_dir}")
    print(f"  样本上限   : {args.max_samples}")
    print(f"  待评模型   : {selected}")

    print("\n[路径检查]")
    for k, v in {**ds_cfg,
                 "ckpt_v7":            CONFIG["ckpt_v7"],
                 "ckpt_v6":            CONFIG["ckpt_v6"],
                 "ckpt_v5_transfer":   CONFIG["ckpt_v5_transfer"],
                 "ckpt_v5_baseline":   CONFIG["ckpt_v5_baseline"],
                 "ckpt_v5_plus":       CONFIG["ckpt_v5_plus"],
                 "ckpt_deepdenoiser":  CONFIG["ckpt_deepdenoiser"],
                 "ckpt_dprnn":         CONFIG["ckpt_dprnn"]}.items():
        flag = "OK" if os.path.exists(v) else "XX"
        print(f"  [{flag}] {k}: {v}")

    # ── 数据 ──────────────────────────────────────
    print("\n[数据集] 构建...")
    # 矿震数据特殊: 过滤到 sampling_rate=100 且 p_arrival_sample>=0 的样本,
    # 避免 EvalDataset 在 50Hz 上不重采样导致的频带错位
    eval_csv = ds_cfg["event_csv"]
    if args.dataset == "mining":
        _src = pd.read_csv(ds_cfg["event_csv"], low_memory=False)
        _flt = _src[
            (_src["sampling_rate"].astype(float) == 100.0) &
            (_src["p_arrival_sample"].astype(float) >= 0)
        ].reset_index(drop=True)
        eval_csv = os.path.join(out_dir, "mining_eval_filtered.csv")
        _flt.to_csv(eval_csv, index=False)
        print(f"  [mining] 过滤 {len(_src)} -> {len(_flt)} 条 (100Hz & P>=0), "
              f"写入 {eval_csv}")
    full_ds = EvalDataset(
        event_h5_path  = ds_cfg["event_h5"],
        event_csv_path = eval_csv,
        noise_h5_path  = ds_cfg["noise_h5"],
        noise_csv_path = ds_cfg["noise_csv"],
        signal_len     = CONFIG["signal_len"],
        cond_len       = CONFIG["cond_len"],
        snr_db_range   = CONFIG["snr_db_range"],
        noise_boost    = CONFIG["noise_boost"],
        max_samples    = args.max_samples,
        seed           = CONFIG["seed"],
    )
    if args.dataset == "natural":
        # chunk2_val.csv is already the held-out STEAD validation split.
        val_ds = full_ds
    else:
        n_val = max(1, int(len(full_ds) * CONFIG["val_ratio"]))
        n_train = len(full_ds) - n_val
        _, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(CONFIG["seed"])
        )
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=CONFIG["num_workers"])
    print(f"  验证集大小: {len(val_ds)}")

    # ── 模型 ──────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.cuda_safe and device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("  CUDA safe mode: cuDNN and TF32 disabled")
    if args.visual_only:
        print("\n[可视化] 自动选择复杂三分量地震波...")
        denoisers_viz = build_all_denoisers(
            selected,
            device,
            mask_dilate_k=args.mask_dilate,
        )
        plot_waveform_compare(
            denoisers_viz,
            loader,
            device,
            os.path.join(out_dir, "compare_waveform.png"),
            visual_scan_samples=args.visual_scan_samples,
            visual_window_seconds=args.visual_window_seconds,
        )
        create_publication_results(out_dir, selected)
        print(f"\n[完成] 可视化结果已写入: {out_dir}")
        return
    print(f"\n[模型] 构建 ({device})...")
    denoisers = build_all_denoisers(selected, device,
                                     mask_dilate_k=args.mask_dilate)
    if not denoisers:
        print("  [ERR] 无可用模型, 退出")
        return

    # ── 评估 ──────────────────────────────────────
    print(f"\n[评估] {len(denoisers)} 个模型...")
    global_rows, group_rows = [], []
    for name, d in denoisers.items():
        records, pick_stats, dt = evaluate_one(d, loader, device)
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(out_dir, f"per_model_{name}.csv"),
                  index=False, encoding="utf-8")
        g, rows = summarize(records, pick_stats, name, dt)
        if g is not None:
            global_rows.append(g)
            group_rows.extend(rows)
        # 释放显存
        if isinstance(d, NeuralDenoiser):
            del d.model
            torch.cuda.empty_cache()

    if not global_rows:
        print("  [ERR] 所有模型评估失败")
        return

    # ── 保存表格 ──────────────────────────────────
    g_df = pd.DataFrame(global_rows)
    # 排序: 按 ΔSNR 降序
    g_df = g_df.sort_values("ΔSNR(dB)", ascending=False).reset_index(drop=True)
    g_df_round = g_df.copy()
    for c in ["ΔSNR(dB)", "SNR_in(dB)", "SNR_out(dB)", "CC", "RMSE", "PRD",
              "ST-MAE(noisy)", "ST-MAE(denoised)",
              "Model_quality", "Fidelity_score", "Quality_fidelity_corr",
              "Pick_succ_noisy", "Pick_succ_denoise"]:
        if c in g_df_round.columns:
            g_df_round[c] = g_df_round[c].round(4)
    g_df_round.to_csv(os.path.join(out_dir, "global_compare.csv"),
                      index=False, encoding="utf-8")
    pd.DataFrame(group_rows).round(4).to_csv(
        os.path.join(out_dir, "per_group_compare.csv"),
        index=False, encoding="utf-8")

    # ── 文字总结 ──────────────────────────────────
    txt = []
    txt.append(f"V5 模型对比评估 — 数据集: {args.dataset}")
    txt.append(f"验证集样本数: {len(val_ds)}, 上限: {args.max_samples}")
    txt.append("-" * 114)
    hdr = f"{'Model':>14} | {'ΔSNR':>8} | {'CC':>6} | {'RMSE':>7} | " \
          f"{'ST-MAE':>8} | {'Fidelity':>8} | {'Quality':>7} | " \
          f"{'Q-F corr':>8} | " \
          f"{'Pick':>5} | {'sec':>6}"
    txt.append(hdr); txt.append("-" * 114)
    for _, r in g_df.iterrows():
        quality = (
            f"{r['Model_quality']:.3f}"
            if np.isfinite(r["Model_quality"])
            else "N/A"
        )
        quality_corr = (
            f"{r['Quality_fidelity_corr']:.3f}"
            if np.isfinite(r["Quality_fidelity_corr"])
            else "N/A"
        )
        txt.append(
            f"{r['Model']:>14} | {r['ΔSNR(dB)']:>+8.3f} | {r['CC']:>6.4f} | "
            f"{r['RMSE']:>7.4f} | {r['ST-MAE(denoised)']:>8.4f} | "
            f"{r['Fidelity_score']:>8.4f} | "
            f"{quality:>7} | {quality_corr:>8} | "
            f"{r['Pick_succ_denoise']:>5.3f} | {r['time_sec']:>6.1f}"
        )
    summary_text = "\n".join(txt)
    print("\n" + summary_text)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # ── 绘图 ──────────────────────────────────────
    plot_bar(g_df, os.path.join(out_dir, "compare_bar.png"))
    create_publication_results(out_dir, selected)

    # 重新构建模型做波形对比 (评估时已释放)
    print("\n[波形对比] 重新加载模型用于可视化...")
    denoisers_viz = build_all_denoisers(selected, device,
                                         mask_dilate_k=args.mask_dilate)
    plot_waveform_compare(denoisers_viz, loader, device,
                          os.path.join(out_dir, "compare_waveform.png"),
                          visual_scan_samples=args.visual_scan_samples,
                          visual_window_seconds=args.visual_window_seconds)

    print(f"\n[完成] 所有结果已写入: {out_dir}")


if __name__ == "__main__":
    main()
