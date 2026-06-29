# -*- coding: utf-8 -*-
"""
v5/dataset_mining.py
====================
LN_mining 矿山微震数据集适配器 (用于 PCD-Net / V6 迁移训练)。

设计目标
--------
* 与 v3.dataset_v3.STEADDatasetV3 的返回字典严格一致, 直接喂给 V6 训练/推理:
    {
      "x":          [3, signal_len]   带噪输入 (clean+scaled_noise)
      "y_clean":    [3, signal_len]   干净目标
      "z_cond":     [3, cond_len]     纯噪声段 (背景噪声条件)
      "valid_mask": [signal_len]      事件有效区掩码 (P 起始后 4000 点)
      "p_onset":    int               裁窗后 P 到时索引
      "has_target": float (1.0)
    }
* 适配 LN_mining 的特点:
    - 波形变长 (15000 ~ 76500), 单条 (T, 3) 存于 data/<trace_name>
    - 采样率 50/100 Hz 混合 -> 统一线性插值到 100 Hz
    - 仅保留有 P 标注 (p_arrival_sample>=0) 的样本以保证监督质量
    - 噪声仍从 part1 chunk1 噪声库采样, 保持与 chunk2 训练一致的噪声协议
"""

from __future__ import annotations
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MiningDatasetV6(Dataset):
    def __init__(
        self,
        # 事件 (LN_mining)
        event_h5_path:  str,
        event_csv_path: str,
        # 噪声 (复用 chunk1)
        noise_h5_path:  str,
        noise_csv_path: str,
        # 兼容 STEADDatasetV3 的占位参数 (V6 当前未启用 Part B)
        raw_h5_path:    str  = None,
        raw_csv_path:   str  = None,
        # 共用参数
        signal_len:     int   = 6000,
        cond_len:       int   = 400,
        snr_range:      tuple = (0.1, 20.0),
        clean_prob:     float = 0.10,
        part_b_ratio:   float = 0.0,
        normalize:      bool  = True,
        seed:           int   = 42,
        debug:          bool  = False,
        # 矿震特有
        target_sr:        float = 100.0,
        require_pick:     bool  = True,   # 只保留 p_arrival_sample>=0
        keep_sr:          tuple = (50.0, 100.0),  # 允许的原始采样率
        pre_event_ratio:  float = 0.2,    # 裁窗时 P 放在窗口前多少处
        eval_mode:        bool  = False,  # True=评估模式, 关闭所有随机性
        snr_for_eval:     float = None,   # 覆盖评估模式下的固定 SNR (线性), None=几何均值
    ):
        super().__init__()
        self.event_h5_path = event_h5_path
        self.noise_h5_path = noise_h5_path
        self.signal_len    = int(signal_len)
        self.cond_len      = int(cond_len)
        self.snr_range     = snr_range
        self.clean_prob    = float(clean_prob)
        self.normalize     = bool(normalize)
        self.debug         = bool(debug)
        self.rng           = np.random.default_rng(seed)
        self.eval_mode     = bool(eval_mode)
        self.target_sr     = float(target_sr)
        self.pre_event_n   = int(self.signal_len * pre_event_ratio)
        # 评估模式使用固定 SNR (snr_range 的几何均值, 或由 snr_for_eval 覆盖)
        _snr_lo, _snr_hi   = snr_range
        if snr_for_eval is not None:
            self._eval_snr = float(snr_for_eval)
        else:
            self._eval_snr = float(np.sqrt(_snr_lo * _snr_hi))

        # ── 事件 CSV 过滤 ───────────────────────────────────
        df = pd.read_csv(event_csv_path, low_memory=False)
        df = df[df['sampling_rate'].astype(float).isin(keep_sr)].reset_index(drop=True)
        if require_pick:
            df = df[df['p_arrival_sample'].astype(float) >= 0].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"[MiningDatasetV6] 过滤后样本为 0: {event_csv_path}")
        self.event_df = df
        self.n_part_a = len(df)
        self.n_part_b = 0
        self.total    = self.n_part_a

        # ── 噪声 CSV (与 STEAD 一致, 字段含 trace_name) ────
        self.noise_df = pd.read_csv(noise_csv_path, low_memory=False)
        if 'trace_name' not in self.noise_df.columns:
            raise RuntimeError("[MiningDatasetV6] 噪声 CSV 缺少 trace_name 列")

        # lazy h5
        self._event_h5 = None
        self._noise_h5 = None

    # ------------------------------------------------------------------
    # h5 lazy open  (DataLoader worker 安全)
    # ------------------------------------------------------------------
    @property
    def event_h5(self):
        if self._event_h5 is None:
            self._event_h5 = h5py.File(self.event_h5_path, 'r',
                                       libver='latest', swmr=True)
        return self._event_h5

    @property
    def noise_h5(self):
        if self._noise_h5 is None:
            self._noise_h5 = h5py.File(self.noise_h5_path, 'r',
                                       libver='latest', swmr=True)
        return self._noise_h5

    def __len__(self):
        return self.total

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _to_3ch(arr: np.ndarray) -> np.ndarray:
        """LN_mining: (T,3) ; STEAD: (T,3). 统一成 [3,T]."""
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[0] == 3 and a.shape[1] != 3:
            return a  # 已经 [3,T]
        if a.shape[1] == 3:
            return a.T
        # 退化: 取前 3 通道或补零
        if a.shape[1] > 3:
            return a[:, :3].T
        out = np.zeros((3, a.shape[0]), dtype=np.float32)
        for c in range(min(3, a.shape[1])):
            out[c] = a[:, c]
        return out

    @staticmethod
    def _resample_linear(wav: np.ndarray, ratio: float) -> np.ndarray:
        """对 [3,T] 沿时间维做线性插值重采样, 输出长度 = round(T*ratio)."""
        if abs(ratio - 1.0) < 1e-6:
            return wav
        C, T = wav.shape
        Tn = max(2, int(round(T * ratio)))
        old_idx = np.linspace(0, T - 1, Tn)
        out = np.empty((C, Tn), dtype=np.float32)
        x_old = np.arange(T, dtype=np.float32)
        for c in range(C):
            out[c] = np.interp(old_idx, x_old, wav[c])
        return out

    def _crop_around(self, wav: np.ndarray, p_idx: int):
        """围绕 p_idx 裁出 [3, signal_len] 窗口, P 放在 pre_event_n 处."""
        C, T = wav.shape
        L = self.signal_len
        if T <= L:
            out = np.zeros((C, L), dtype=np.float32)
            out[:, :T] = wav
            return out, max(0, min(p_idx, L - 1))
        # 期望窗口起点
        start = p_idx - self.pre_event_n
        # 训练时加随机抖动；评估时不抖动，保证可复现
        if not self.eval_mode:
            start += int(self.rng.integers(-self.pre_event_n // 4,
                                           self.pre_event_n // 4 + 1))
        start = int(max(0, min(start, T - L)))
        out = wav[:, start:start + L].copy()
        new_p = max(0, min(p_idx - start, L - 1))
        return out, new_p

    def _normalize(self, wav: np.ndarray) -> np.ndarray:
        m = np.abs(wav).max()
        return wav / m if m > 1e-10 else wav

    def _mix_snr(self, signal, noise, snr_linear: float):
        sp = np.mean(signal ** 2)
        npw = np.mean(noise ** 2)
        if sp < 1e-10 or npw < 1e-10:
            return signal.copy()
        scale = float(np.clip(np.sqrt(sp / (snr_linear * npw)), 0, 10))
        return signal + scale * noise

    # ------------------------------------------------------------------
    # 加载 / 主流程
    # ------------------------------------------------------------------
    def _load_event(self, trace_name: str, orig_sr: float):
        ds = self.event_h5['data'][trace_name][()]
        wav = self._to_3ch(ds)
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        ratio = self.target_sr / float(orig_sr)
        if abs(ratio - 1.0) > 1e-6:
            wav = self._resample_linear(wav, ratio)
        return wav, ratio

    def _load_noise(self, trace_name: str):
        ds = self.noise_h5['data'][trace_name][()]
        wav = self._to_3ch(ds)
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        # 噪声若过短, tile 补齐
        if wav.shape[1] < self.signal_len:
            reps = int(np.ceil(self.signal_len / max(1, wav.shape[1])))
            wav = np.tile(wav, (1, reps))[:, :self.signal_len]
        else:
            if self.eval_mode:
                # 评估模式: 固定从头裁段
                start = 0
            else:
                # 训练模式: 随机起点
                start = int(self.rng.integers(0, wav.shape[1] - self.signal_len + 1))
            wav = wav[:, start:start + self.signal_len].copy()
        return wav

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            if self.debug:
                print(f"[MiningDatasetV6] idx={idx} skip: {e}")
            return self._zero_sample()

    def _get(self, idx):
        row = self.event_df.iloc[idx]
        orig_sr = float(row.get('sampling_rate', self.target_sr) or self.target_sr)
        wav_full, ratio = self._load_event(row['trace_name'], orig_sr)

        # P 到时映射到 target_sr 索引
        p_raw = int(row['p_arrival_sample'])
        p_full = int(round(p_raw * ratio))

        wave_clean, p_onset = self._crop_around(wav_full, p_full)

        # 噪声采样: 评估模式按 idx 确定性对应, 训练模式随机
        if self.eval_mode:
            ni = idx % len(self.noise_df)
        else:
            ni = int(self.rng.integers(0, len(self.noise_df)))
        n_name = self.noise_df.iloc[ni]['trace_name']
        noise_wave = self._load_noise(n_name)

        if self.normalize:
            wave_clean = self._normalize(wave_clean)
            noise_wave = self._normalize(noise_wave)

        # z_cond: 取归一化噪声前 cond_len 段
        noise_cond = noise_wave[:, :self.cond_len].copy()
        nc_scale = np.abs(noise_cond).max()
        if nc_scale > 1e-10:
            noise_cond = noise_cond / nc_scale

        # mix: 评估模式使用固定 SNR, 不掷 clean_prob 骰子
        if self.eval_mode:
            x_noisy = self._mix_snr(wave_clean, noise_wave, self._eval_snr)
        elif self.rng.random() < self.clean_prob:
            x_noisy = wave_clean.copy()
        else:
            snr = float(self.rng.uniform(*self.snr_range))
            x_noisy = self._mix_snr(wave_clean, noise_wave, snr)

        y_clean = wave_clean

        # valid_mask: P 起始后 4000 点为有效区
        valid_mask = np.zeros(self.signal_len, dtype=np.float32)
        end = min(p_onset + 4000, self.signal_len)
        valid_mask[p_onset:end] = 1.0

        return {
            "x":          torch.from_numpy(np.clip(x_noisy,    -10, 10)).float(),
            "y_clean":    torch.from_numpy(np.clip(y_clean,    -10, 10)).float(),
            "z_cond":     torch.from_numpy(np.clip(noise_cond, -10, 10)).float(),
            "valid_mask": torch.from_numpy(valid_mask),
            "p_onset":    torch.tensor(p_onset, dtype=torch.long),
            "has_target": torch.tensor(1.0, dtype=torch.float32),
        }

    def _zero_sample(self):
        return {
            "x":          torch.zeros(3, self.signal_len),
            "y_clean":    torch.zeros(3, self.signal_len),
            "z_cond":     torch.zeros(3, self.cond_len),
            "valid_mask": torch.zeros(self.signal_len),
            "p_onset":    torch.tensor(0, dtype=torch.long),
            "has_target": torch.tensor(0.0, dtype=torch.float32),
        }


# ============================================================
#  独立自测
# ============================================================
if __name__ == "__main__":
    ds = MiningDatasetV6(
        event_h5_path  = r"D:/X/part2/data/LN_mining.hdf5",
        event_csv_path = r"D:/X/part2/data/LN_mining.csv",
        noise_h5_path  = r"D:/X/p_wave/data/chunk1.hdf5",
        noise_csv_path = r"D:/X/p_wave/data/chunk1.csv",
        signal_len = 6000, cond_len = 400,
        require_pick = True, debug = True,
    )
    print(f"[OK] N = {len(ds)}")
    s = ds[0]
    for k, v in s.items():
        print(f"  {k}: {tuple(v.shape) if hasattr(v,'shape') else v}")
