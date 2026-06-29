# # -*- coding: utf-8 -*-
# # D:\X\denoise\part1\v3\transfer\val_wavelet.py
# """
# 传统小波去噪对比实验
# - 数据流与 val_addnoise.py 完全一致（chunk2后1000条，加噪SNR -10~0dB）
# - 用 PyWavelets 对 noisy 做小波软阈值去噪
# - 输出同结构的 metrics.csv / summary.json / SVG三列图
# """
#
# import os
# import sys
# import json
# import h5py
# import numpy as np
# import pandas as pd
# import pywt
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# matplotlib.rcParams["font.family"] = "Times New Roman"
# matplotlib.rcParams["axes.unicode_minus"] = False
#
# # ============================================================
# # 配置（与 val_addnoise.py 保持一致）
# # ============================================================
# CONFIG = {
#     "event_h5":  r"D:/X/p_wave/data/chunk2.hdf5",
#     "event_csv": r"D:/X/p_wave/data/chunk2.csv",
#     "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
#     "noise_csv": r"D:/X/p_wave/data/chunk1.csv",
#
#     "use_tail_n":  1000,
#     "max_samples": 200,
#
#     "snr_db_range": (-10.0, 0.0),
#     "fixed_snr_db": None,
#     "noise_boost":  1.0,
#
#     "signal_len": 6000,
#     "cond_len":   400,
#     "seed":       42,
#     "fs":         100,   # Hz
#
#     # 小波参数
#     "wavelet":    "db4",   # 小波基
#     "level":      5,       # 分解层数
#     "threshold_mode": "soft",   # soft / hard
#
#     # 输出
#     "out_dir":      r"v3/val_wavelet_outputs",
#     "out_h5":       r"v3/val_wavelet_outputs/clean_noisy_denoised.hdf5",
#     "metrics_csv":  r"v3/val_wavelet_outputs/metrics.csv",
#     "summary_json": r"v3/val_wavelet_outputs/summary.json",
#     "save_svg_num": 50,
#     "svg_dir":      r"v3/val_wavelet_outputs/svg_triplets",
# }
#
# # ============================================================
# # 小波去噪核心
# # ============================================================
# def wavelet_denoise_1d(signal: np.ndarray, wavelet: str, level: int, mode: str) -> np.ndarray:
#     """对单通道 1D 信号做小波软/硬阈值去噪"""
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     # 用最细节层估计噪声标准差（MAD估计）
#     detail_finest = coeffs[-1]
#     sigma = np.median(np.abs(detail_finest)) / 0.6745
#     threshold = sigma * np.sqrt(2 * np.log(len(signal)))
#
#     # 对所有细节层做阈值
#     coeffs_thresh = [coeffs[0]]  # 近似系数不处理
#     for c in coeffs[1:]:
#         coeffs_thresh.append(pywt.threshold(c, threshold, mode=mode))
#
#     return pywt.waverec(coeffs_thresh, wavelet)[:len(signal)].astype(np.float32)
#
# def wavelet_denoise_3ch(wave_3t: np.ndarray, wavelet: str, level: int, mode: str) -> np.ndarray:
#     """对 [3, T] 三通道波形逐通道去噪"""
#     out = np.zeros_like(wave_3t)
#     for c in range(3):
#         out[c] = wavelet_denoise_1d(wave_3t[c], wavelet, level, mode)
#     return out
#
# # ============================================================
# # 数据加载（复用 val_addnoise 的逻辑，不依赖 torch）
# # ============================================================
# def load_wave(h5f, trace_name, signal_len):
#     x = h5f["data"][trace_name][:]
#     x = x.T.astype(np.float32)          # [3, T]
#     x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#     T = x.shape[1]
#     if T >= signal_len:
#         return x[:, :signal_len]
#     out = np.zeros((3, signal_len), dtype=np.float32)
#     out[:, :T] = x
#     return out
#
# def normalize_peak(x):
#     m = np.abs(x).max()
#     return (x / m, m) if m > 1e-10 else (x, 1.0)
#
# def mix_snr_db(clean_n, noise_n, snr_db):
#     snr_lin = 10.0 ** (snr_db / 10.0)
#     ps = np.mean(clean_n ** 2)
#     pn = np.mean(noise_n ** 2)
#     if ps < 1e-12 or pn < 1e-12:
#         return clean_n.copy(), 0.0
#     scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
#     return clean_n + scale * noise_n, scale
#
# def get_p_onset(row, signal_len):
#     for col in ["p_arrival_sample", "p_onset", "itp"]:
#         if col in row.index and not pd.isna(row[col]):
#             try:
#                 v = int(row[col])
#                 if 0 <= v < signal_len:
#                     return v
#             except Exception:
#                 pass
#     return signal_len // 10
#
# # ============================================================
# # SNR 计算
# # ============================================================
# def compute_snr_db(clean, residual, p_onset, signal_len, cond_len):
#     """clean/residual: [3,T]，在P波后4000采样窗口内计算"""
#     ev_l = p_onset
#     ev_r = min(signal_len, p_onset + 4000)
#     if ev_r - ev_l < 10:
#         ev_r = min(signal_len, ev_l + 1000)
#     sig = np.mean(clean[:, ev_l:ev_r] ** 2) + 1e-12
#     noi = np.mean(residual[:, ev_l:ev_r] ** 2) + 1e-12
#     return float(np.clip(10.0 * np.log10(sig / noi), -50, 50))
#
# # ============================================================
# # SVG 三列图（与 val_addnoise 一致）
# # ============================================================
# def save_triplet_svg(clean_3t, noisy_3t, deno_3t, out_svg, title="", fs=None):
#     os.makedirs(os.path.dirname(out_svg), exist_ok=True)
#     ch_names = ["E", "N", "Z"]
#     T = clean_3t.shape[-1]
#     t = np.arange(T) if fs is None else np.arange(T) / float(fs)
#     xlab = "sample" if fs is None else "time (s)"
#
#     fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
#     for j, col_title in enumerate(["Clean", "Noisy", "Denoised (Wavelet)"]):
#         axes[0, j].set_title(col_title, fontsize=11)
#
#     for i in range(3):
#         c, n, d = clean_3t[i], noisy_3t[i], deno_3t[i]
#         ymax = max(np.max(np.abs(c)), np.max(np.abs(n)), np.max(np.abs(d)), 1e-6)
#         for j, sig in enumerate([c, n, d]):
#             axes[i, j].plot(t, sig, lw=0.8, color="#1f77b4")
#             axes[i, j].set_ylim(-ymax, ymax)
#             axes[i, j].grid(alpha=0.25, linestyle="--")
#             if j == 0:
#                 axes[i, j].set_ylabel(ch_names[i])
#
#     for j in range(3):
#         axes[2, j].set_xlabel(xlab)
#
#     fig.suptitle(title, fontsize=12)
#     fig.tight_layout()
#     fig.savefig(out_svg, format="svg")
#     plt.close(fig)
#
# # ============================================================
# # 主流程
# # ============================================================
# def main():
#     np.random.seed(CONFIG["seed"])
#     os.makedirs(CONFIG["out_dir"], exist_ok=True)
#     os.makedirs(CONFIG["svg_dir"], exist_ok=True)
#
#     for k in ["event_h5", "event_csv", "noise_h5", "noise_csv"]:
#         if not os.path.exists(CONFIG[k]):
#             raise FileNotFoundError(f"{k} not found: {CONFIG[k]}")
#
#     event_df = pd.read_csv(CONFIG["event_csv"], low_memory=False)
#     noise_df = pd.read_csv(CONFIG["noise_csv"], low_memory=False)
#
#     if CONFIG["use_tail_n"] is not None:
#         event_df = event_df.tail(int(CONFIG["use_tail_n"])).reset_index(drop=True)
#     if CONFIG["max_samples"] is not None:
#         event_df = event_df.iloc[:int(CONFIG["max_samples"])].reset_index(drop=True)
#
#     print(f"[INFO] events={len(event_df)}, noises={len(noise_df)}")
#
#     metrics = []
#     svg_saved = 0
#     used_names = set()
#
#     with h5py.File(CONFIG["event_h5"], "r") as ev_h5, \
#          h5py.File(CONFIG["noise_h5"], "r") as no_h5, \
#          h5py.File(CONFIG["out_h5"], "w") as out_h5:
#
#         g_clean = out_h5.create_group("clean")
#         g_noisy = out_h5.create_group("noisy")
#         g_deno  = out_h5.create_group("denoised")
#
#         for idx, row in event_df.iterrows():
#             trace_name = str(row["trace_name"])
#             rng = np.random.default_rng(CONFIG["seed"] + idx)
#
#             # clean
#             clean = load_wave(ev_h5, trace_name, CONFIG["signal_len"])
#             clean_n, _ = normalize_peak(clean)
#
#             # noise
#             ni = int(rng.integers(0, len(noise_df)))
#             noise_name = str(noise_df.iloc[ni]["trace_name"])
#             noise = load_wave(no_h5, noise_name, CONFIG["signal_len"])
#             noise_n, _ = normalize_peak(noise)
#
#             # SNR混合
#             if CONFIG["fixed_snr_db"] is None:
#                 snr_db = float(rng.uniform(CONFIG["snr_db_range"][0], CONFIG["snr_db_range"][1]))
#             else:
#                 snr_db = float(CONFIG["fixed_snr_db"])
#
#             noisy_base, _ = mix_snr_db(clean_n, noise_n, snr_db)
#             noisy_n = clean_n + CONFIG["noise_boost"] * (noisy_base - clean_n)
#             noisy_n = np.clip(noisy_n, -10, 10).astype(np.float32)
#
#             # 小波去噪
#             deno = wavelet_denoise_3ch(
#                 noisy_n,
#                 wavelet=CONFIG["wavelet"],
#                 level=CONFIG["level"],
#                 mode=CONFIG["threshold_mode"],
#             )
#
#             # 指标
#             p_onset = get_p_onset(row, CONFIG["signal_len"])
#             snr_in  = compute_snr_db(clean_n, noisy_n - clean_n, p_onset, CONFIG["signal_len"], CONFIG["cond_len"])
#             snr_out = compute_snr_db(clean_n, deno - clean_n,    p_onset, CONFIG["signal_len"], CONFIG["cond_len"])
#
#             # 唯一名
#             base = trace_name.replace("/", "_").replace("\\", "_")
#             uname = base
#             if uname in used_names:
#                 i = 1
#                 while f"{base}_{i}" in used_names:
#                     i += 1
#                 uname = f"{base}_{i}"
#             used_names.add(uname)
#
#             # H5
#             g_clean.create_dataset(uname, data=clean_n.T.astype(np.float32), compression="gzip")
#             g_noisy.create_dataset(uname, data=noisy_n.T.astype(np.float32), compression="gzip")
#             g_deno.create_dataset(uname,  data=deno.T.astype(np.float32),    compression="gzip")
#
#             metrics.append({
#                 "trace_name":    trace_name,
#                 "noise_trace":   noise_name,
#                 "snr_set_db":    snr_db,
#                 "input_snr_db":  snr_in,
#                 "output_snr_db": snr_out,
#                 "snr_gain_db":   snr_out - snr_in,
#             })
#
#             # SVG
#             if svg_saved < CONFIG["save_svg_num"]:
#                 out_svg = os.path.join(CONFIG["svg_dir"], f"{uname}_triplet.svg")
#                 save_triplet_svg(
#                     clean_n, noisy_n, deno,
#                     out_svg=out_svg,
#                     title=f"{trace_name} | SNR_set={snr_db:.2f} dB | SNR_gain={snr_out - snr_in:.2f} dB",
#                     fs=CONFIG["fs"],
#                 )
#                 svg_saved += 1
#
#             if (idx + 1) % 50 == 0:
#                 print(f"  processed {idx + 1}/{len(event_df)}")
#
#     mdf = pd.DataFrame(metrics)
#     mdf.to_csv(CONFIG["metrics_csv"], index=False)
#
#     summary = {
#         "method": f"wavelet ({CONFIG['wavelet']}, level={CONFIG['level']}, {CONFIG['threshold_mode']})",
#         "n_samples":          int(len(mdf)),
#         "snr_set_db_mean":    float(mdf["snr_set_db"].mean()),
#         "input_snr_db_mean":  float(mdf["input_snr_db"].mean()),
#         "output_snr_db_mean": float(mdf["output_snr_db"].mean()),
#         "snr_gain_db_mean":   float(mdf["snr_gain_db"].mean()),
#         "snr_gain_db_median": float(mdf["snr_gain_db"].median()),
#         "svg_saved":          int(svg_saved),
#         "out_h5":             CONFIG["out_h5"],
#         "metrics_csv":        CONFIG["metrics_csv"],
#     }
#
#     with open(CONFIG["summary_json"], "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)
#
#     print("\n========== 小波去噪完成 ==========")
#     print(json.dumps(summary, indent=2, ensure_ascii=False))
#
# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
# D:\X\denoise\part1\v3\transfer\val_wavelet.py
"""
传统小波去噪对比实验
输出指标：ΔSNR / RMSE / ST-MAE / PRD / 参数量(N/A) / 推理时间
"""

import os
import time
import json
import h5py
import numpy as np
import pandas as pd
import pywt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "event_h5":  r"D:/X/p_wave/data/chunk2.hdf5",
    "event_csv": r"D:/X/p_wave/data/chunk2.csv",
    "noise_h5":  r"D:/X/p_wave/data/chunk1.hdf5",
    "noise_csv": r"D:/X/p_wave/data/chunk1.csv",

    "use_tail_n":  1000,
    "max_samples": 200,

    "snr_db_range": (-10.0, 0.0),
    "fixed_snr_db": None,
    "noise_boost":  1.0,

    "signal_len": 6000,
    "cond_len":   400,
    "seed":       42,
    "fs":         100,

    "wavelet":        "db4",
    "level":          5,
    "threshold_mode": "soft",

    "out_dir":      r"v3/val_wavelet_outputs",
    "out_h5":       r"v3/val_wavelet_outputs/clean_noisy_denoised.hdf5",
    "metrics_csv":  r"v3/val_wavelet_outputs/metrics.csv",
    "summary_json": r"v3/val_wavelet_outputs/summary.json",
    "save_svg_num": 50,
    "svg_dir":      r"v3/val_wavelet_outputs/svg_triplets",
}

# ============================================================
# 小波去噪核心
# ============================================================
def wavelet_denoise_1d(signal, wavelet, level, mode):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode=mode))
    return pywt.waverec(coeffs_thresh, wavelet)[:len(signal)].astype(np.float32)

def wavelet_denoise_3ch(wave_3t, wavelet, level, mode):
    out = np.zeros_like(wave_3t)
    for c in range(3):
        out[c] = wavelet_denoise_1d(wave_3t[c], wavelet, level, mode)
    return out

# ============================================================
# 数据加载
# ============================================================
def load_wave(h5f, trace_name, signal_len):
    x = h5f["data"][trace_name][:]
    x = x.T.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    T = x.shape[1]
    if T >= signal_len:
        return x[:, :signal_len]
    out = np.zeros((3, signal_len), dtype=np.float32)
    out[:, :T] = x
    return out

def normalize_peak(x):
    m = np.abs(x).max()
    return (x / m, m) if m > 1e-10 else (x, 1.0)

def mix_snr_db(clean_n, noise_n, snr_db):
    snr_lin = 10.0 ** (snr_db / 10.0)
    ps = np.mean(clean_n ** 2)
    pn = np.mean(noise_n ** 2)
    if ps < 1e-12 or pn < 1e-12:
        return clean_n.copy(), 0.0
    scale = float(np.clip(np.sqrt(ps / (snr_lin * pn)), 0.0, 10.0))
    return clean_n + scale * noise_n, scale

def get_p_onset(row, signal_len):
    for col in ["p_arrival_sample", "p_onset", "itp"]:
        if col in row.index and not pd.isna(row[col]):
            try:
                v = int(row[col])
                if 0 <= v < signal_len:
                    return v
            except Exception:
                pass
    return signal_len // 10

# ============================================================
# 指标计算
# ============================================================
def compute_snr_db(clean, residual, p_onset, signal_len):
    ev_l = p_onset
    ev_r = min(signal_len, p_onset + 4000)
    if ev_r - ev_l < 10:
        ev_r = min(signal_len, ev_l + 1000)
    sig = np.mean(clean[:, ev_l:ev_r] ** 2) + 1e-12
    noi = np.mean(residual[:, ev_l:ev_r] ** 2) + 1e-12
    return float(np.clip(10.0 * np.log10(sig / noi), -50, 50))

def compute_rmse(clean, denoised, p_onset, signal_len):
    """P波后4000采样窗口内的均方根误差（三通道平均）"""
    ev_l = p_onset
    ev_r = min(signal_len, p_onset + 4000)
    if ev_r - ev_l < 10:
        ev_r = min(signal_len, ev_l + 1000)
    diff = clean[:, ev_l:ev_r] - denoised[:, ev_l:ev_r]
    return float(np.sqrt(np.mean(diff ** 2)))

def compute_prd(clean, denoised, p_onset, signal_len):
    """
    PRD (%) = sqrt( sum((clean-denoised)^2) / sum(clean^2) ) × 100
    在P波窗口内计算，三通道联合
    """
    ev_l = p_onset
    ev_r = min(signal_len, p_onset + 4000)
    if ev_r - ev_l < 10:
        ev_r = min(signal_len, ev_l + 1000)
    c = clean[:, ev_l:ev_r]
    d = denoised[:, ev_l:ev_r]
    num = np.sum((c - d) ** 2)
    den = np.sum(c ** 2) + 1e-12
    return float(np.sqrt(num / den) * 100.0)

def compute_st_mae(clean_1d, denoised_1d, fs, win_ms=100.0, overlap=0.5):
    """
    滑动窗口MAE（单通道），返回所有窗口的MAE均值作为标量
    win_ms : 窗长（毫秒）
    """
    win_len = int(fs * win_ms / 1000)
    hop_len = max(1, int(win_len * (1 - overlap)))
    T = len(clean_1d)
    maes = []
    start = 0
    while start + win_len <= T:
        end = start + win_len
        maes.append(np.abs(clean_1d[start:end] - denoised_1d[start:end]).mean())
        start += hop_len
    return float(np.mean(maes)) if maes else 0.0

def compute_st_mae_3ch(clean, denoised, fs, win_ms=100.0, overlap=0.5):
    """三通道ST-MAE均值"""
    vals = [compute_st_mae(clean[c], denoised[c], fs, win_ms, overlap)
            for c in range(3)]
    return float(np.mean(vals))

# ============================================================
# SVG 三列图
# ============================================================
def save_triplet_svg(clean_3t, noisy_3t, deno_3t, out_svg, title="", fs=None):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    ch_names = ["E", "N", "Z"]
    T = clean_3t.shape[-1]
    t = np.arange(T) if fs is None else np.arange(T) / float(fs)
    xlab = "sample" if fs is None else "time (s)"

    fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    for j, col_title in enumerate(["Clean", "Noisy", "Denoised (Wavelet)"]):
        axes[0, j].set_title(col_title, fontsize=11)

    for i in range(3):
        c, n, d = clean_3t[i], noisy_3t[i], deno_3t[i]
        ymax = max(np.max(np.abs(c)), np.max(np.abs(n)), np.max(np.abs(d)), 1e-6)
        for j, sig in enumerate([c, n, d]):
            axes[i, j].plot(t, sig, lw=0.8, color="#1f77b4")
            axes[i, j].set_ylim(-ymax, ymax)
            axes[i, j].grid(alpha=0.25, linestyle="--")
            if j == 0:
                axes[i, j].set_ylabel(ch_names[i])
    for j in range(3):
        axes[2, j].set_xlabel(xlab)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)

# ============================================================
# 主流程
# ============================================================
def main():
    np.random.seed(CONFIG["seed"])
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    os.makedirs(CONFIG["svg_dir"], exist_ok=True)

    for k in ["event_h5", "event_csv", "noise_h5", "noise_csv"]:
        if not os.path.exists(CONFIG[k]):
            raise FileNotFoundError(f"{k} not found: {CONFIG[k]}")

    event_df = pd.read_csv(CONFIG["event_csv"], low_memory=False)
    noise_df = pd.read_csv(CONFIG["noise_csv"], low_memory=False)

    if CONFIG["use_tail_n"] is not None:
        event_df = event_df.tail(int(CONFIG["use_tail_n"])).reset_index(drop=True)
    if CONFIG["max_samples"] is not None:
        event_df = event_df.iloc[:int(CONFIG["max_samples"])].reset_index(drop=True)

    print(f"[INFO] events={len(event_df)}, noises={len(noise_df)}")

    metrics  = []
    svg_saved = 0
    used_names = set()
    infer_times = []   # 推理时间列表（ms）

    with h5py.File(CONFIG["event_h5"], "r") as ev_h5, \
         h5py.File(CONFIG["noise_h5"], "r") as no_h5, \
         h5py.File(CONFIG["out_h5"], "w") as out_h5:

        g_clean = out_h5.create_group("clean")
        g_noisy = out_h5.create_group("noisy")
        g_deno  = out_h5.create_group("denoised")

        for idx, row in event_df.iterrows():
            trace_name = str(row["trace_name"])
            rng = np.random.default_rng(CONFIG["seed"] + idx)

            # ── 数据准备 ─────────────────────────────────
            clean = load_wave(ev_h5, trace_name, CONFIG["signal_len"])
            clean_n, _ = normalize_peak(clean)

            ni = int(rng.integers(0, len(noise_df)))
            noise_name = str(noise_df.iloc[ni]["trace_name"])
            noise = load_wave(no_h5, noise_name, CONFIG["signal_len"])
            noise_n, _ = normalize_peak(noise)

            if CONFIG["fixed_snr_db"] is None:
                snr_db = float(rng.uniform(CONFIG["snr_db_range"][0], CONFIG["snr_db_range"][1]))
            else:
                snr_db = float(CONFIG["fixed_snr_db"])

            noisy_base, _ = mix_snr_db(clean_n, noise_n, snr_db)
            noisy_n = clean_n + CONFIG["noise_boost"] * (noisy_base - clean_n)
            noisy_n = np.clip(noisy_n, -10, 10).astype(np.float32)

            # ── 小波去噪（计时）─────────────────────────
            t0 = time.perf_counter()
            deno = wavelet_denoise_3ch(
                noisy_n,
                wavelet=CONFIG["wavelet"],
                level=CONFIG["level"],
                mode=CONFIG["threshold_mode"],
            )
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0   # 转为毫秒
            infer_times.append(infer_ms)

            # ── 指标计算 ──────────────────────────────────
            p_onset = get_p_onset(row, CONFIG["signal_len"])

            snr_in  = compute_snr_db(clean_n, noisy_n - clean_n,
                                     p_onset, CONFIG["signal_len"])
            snr_out = compute_snr_db(clean_n, deno - clean_n,
                                     p_onset, CONFIG["signal_len"])
            rmse    = compute_rmse(clean_n, deno, p_onset, CONFIG["signal_len"])
            prd     = compute_prd(clean_n, deno, p_onset, CONFIG["signal_len"])
            st_mae  = compute_st_mae_3ch(clean_n, deno,
                                         fs=CONFIG["fs"],
                                         win_ms=100.0, overlap=0.5)

            # ── 唯一名 & H5存储 ───────────────────────────
            base  = trace_name.replace("/", "_").replace("\\", "_")
            uname = base
            if uname in used_names:
                i = 1
                while f"{base}_{i}" in used_names:
                    i += 1
                uname = f"{base}_{i}"
            used_names.add(uname)

            g_clean.create_dataset(uname, data=clean_n.T.astype(np.float32), compression="gzip")
            g_noisy.create_dataset(uname, data=noisy_n.T.astype(np.float32), compression="gzip")
            g_deno.create_dataset(uname,  data=deno.T.astype(np.float32),    compression="gzip")

            metrics.append({
                "trace_name":    trace_name,
                "noise_trace":   noise_name,
                "snr_set_db":    snr_db,
                "input_snr_db":  snr_in,
                "output_snr_db": snr_out,
                "snr_gain_db":   snr_out - snr_in,
                "rmse":          rmse,
                "prd_pct":       prd,
                "st_mae":        st_mae,
                "infer_ms":      infer_ms,
            })

            # ── SVG ───────────────────────────────────────
            if svg_saved < CONFIG["save_svg_num"]:
                out_svg = os.path.join(CONFIG["svg_dir"], f"{uname}_triplet.svg")
                save_triplet_svg(
                    clean_n, noisy_n, deno,
                    out_svg=out_svg,
                    title=(f"{trace_name} | SNR_set={snr_db:.2f} dB | "
                           f"ΔSNR={snr_out - snr_in:.2f} dB | "
                           f"RMSE={rmse:.5f} | PRD={prd:.2f}%"),
                    fs=CONFIG["fs"],
                )
                svg_saved += 1

            if (idx + 1) % 50 == 0:
                print(f"  processed {idx + 1}/{len(event_df)}")

    # ── 汇总 ──────────────────────────────────────────────
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(CONFIG["metrics_csv"], index=False)

    summary = {
        "method": (f"wavelet ({CONFIG['wavelet']}, "
                   f"level={CONFIG['level']}, {CONFIG['threshold_mode']})"),
        "n_samples":           int(len(mdf)),
        "snr_set_db_mean":     float(mdf["snr_set_db"].mean()),
        "input_snr_db_mean":   float(mdf["input_snr_db"].mean()),
        "output_snr_db_mean":  float(mdf["output_snr_db"].mean()),
        "delta_snr_db_mean":   float(mdf["snr_gain_db"].mean()),
        "delta_snr_db_median": float(mdf["snr_gain_db"].median()),
        "rmse_mean":           float(mdf["rmse"].mean()),
        "prd_pct_mean":        float(mdf["prd_pct"].mean()),
        "st_mae_mean":         float(mdf["st_mae"].mean()),
        "infer_ms_mean":       float(mdf["infer_ms"].mean()),
        "infer_ms_median":     float(mdf["infer_ms"].median()),
        "params_M":            "N/A (traditional method)",
        "svg_saved":           int(svg_saved),
        "out_h5":              CONFIG["out_h5"],
        "metrics_csv":         CONFIG["metrics_csv"],
    }

    with open(CONFIG["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 终端打印对比表 ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  小波去噪指标汇总（可直接填入对比表）")
    print("=" * 72)
    print(f"  {'指标':<22} {'值':>15}")
    print(f"  {'-'*22} {'-'*15}")
    print(f"  {'ΔSNR (dB)':<22} {summary['delta_snr_db_mean']:>15.4f}")
    print(f"  {'RMSE':<22} {summary['rmse_mean']:>15.6f}")
    print(f"  {'ST-MAE':<22} {summary['st_mae_mean']:>15.6f}")
    print(f"  {'平均PRD (%)':<22} {summary['prd_pct_mean']:>15.4f}")
    print(f"  {'参数量 (M)':<22} {'N/A':>15}")
    print(f"  {'推理时间 (ms)':<22} {summary['infer_ms_mean']:>15.3f}")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()