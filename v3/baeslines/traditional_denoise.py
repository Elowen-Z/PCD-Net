# v3/compare/traditional_denoise.py
"""
传统去噪方法实现：
  1. BPF  : 6 阶 Butterworth 带通滤波器（根据信号主频自适应设计）
  2. Wavelet : Daubechies 小波软阈值降噪
"""

import numpy as np
from scipy import signal as sp_signal
import pywt


# ============================================================
#  1. BPF：6 阶 Butterworth 带通滤波器
# ============================================================

def estimate_dominant_freq(wave: np.ndarray, fs: float) -> tuple:
    """
    估计信号主频 → 自动设计带通范围

    wave : [3, T] 三分量波形
    fs   : 采样率（Hz）

    return : (f_low, f_high)  Hz
    """
    # 用 Z 分量（通道2）估计主频
    z = wave[2]
    N = len(z)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    power = np.abs(np.fft.rfft(z)) ** 2

    # 只看 0.5~45 Hz 范围（地震波典型频段）
    valid = (freqs >= 0.5) & (freqs <= 45.0)
    if valid.sum() == 0:
        return 1.0, 20.0

    power_valid = power[valid]
    freqs_valid = freqs[valid]

    # 主频 = 功率加权中心频率
    f_dominant = np.sum(freqs_valid * power_valid) / (
            np.sum(power_valid) + 1e-10
    )
    f_dominant = np.clip(f_dominant, 1.0, 40.0)

    # 带通范围：主频的 1/3 ~ 3 倍，再限制到合理范围
    f_low = max(0.5, f_dominant / 3.0)
    f_high = min(fs / 2.0 - 1.0, f_dominant * 3.0)

    # 保证 f_low < f_high 且有足够带宽
    if f_high - f_low < 1.0:
        f_low = max(0.5, f_dominant - 2.0)
        f_high = min(fs / 2.0 - 1.0, f_dominant + 2.0)

    return float(f_low), float(f_high)


def butterworth_bandpass(wave: np.ndarray,
                         fs: float = 100.0,
                         f_low: float = None,
                         f_high: float = None,
                         order: int = 6,
                         adaptive: bool = True) -> np.ndarray:
    """
    6 阶 Butterworth 带通滤波器

    wave     : [3, T]  归一化波形
    fs       : 采样率 Hz
    f_low    : 低截止频率 Hz（None → 自适应估计）
    f_high   : 高截止频率 Hz（None → 自适应估计）
    order    : 滤波器阶数，默认 6
    adaptive : True → 对每条波形自适应估计主频

    return   : [3, T]  滤波后波形
    """
    out = np.zeros_like(wave)

    for ch in range(wave.shape[0]):
        x = wave[ch]

        # 自适应主频估计
        if adaptive or f_low is None or f_high is None:
            fl, fh = estimate_dominant_freq(wave, fs)
        else:
            fl, fh = f_low, f_high

        # 归一化截止频率（Nyquist = fs/2）
        nyq = fs / 2.0
        wl = fl / nyq
        wh = fh / nyq

        # 数值安全检查
        wl = np.clip(wl, 1e-4, 0.99)
        wh = np.clip(wh, 1e-4, 0.99)
        if wl >= wh:
            wl = wh * 0.5

        try:
            # 使用 sosfilt（比 lfilter 数值更稳定）
            sos = sp_signal.butter(
                order, [wl, wh],
                btype='bandpass',
                output='sos'
            )
            out[ch] = sp_signal.sosfiltfilt(sos, x)
        except Exception:
            out[ch] = x  # 滤波失败则返回原始

    # 归一化（保持与模型输出一致）
    m = np.abs(out).max()
    if m > 1e-10:
        out = out / m

    return out.astype(np.float32)


# ============================================================
#  2. Wavelet：Daubechies 软阈值降噪
# ============================================================

def estimate_noise_std(coeffs: np.ndarray) -> float:
    """
    用最细节层系数估计噪声标准差（MAD 估计器）
    鲁棒性好，不受信号幅度影响
    """
    return float(np.median(np.abs(coeffs)) / 0.6745)


def wavelet_denoise_channel(x: np.ndarray,
                            wavelet: str = 'db4',
                            level: int = None,
                            threshold_mode: str = 'soft',
                            threshold_scale: float = 1.0) -> np.ndarray:
    """
    单通道小波软阈值降噪

    x               : [T] 单通道波形
    wavelet         : 小波基，默认 db4（Daubechies 4阶）
    level           : 分解层数，None → 自动（log2(T)）
    threshold_mode  : 'soft' 或 'hard'
    threshold_scale : 阈值缩放因子（>1 → 更激进去噪）
    """
    T = len(x)

    # 自动确定分解层数
    if level is None:
        max_level = pywt.dwt_max_level(T, wavelet)
        level = min(max_level, 8)  # 最多8层

    # 小波分解
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # 用最细节层估计噪声
    sigma = estimate_noise_std(coeffs[-1])

    # 通用阈值（Universal Threshold）
    # thr = sigma * sqrt(2 * log(N))
    thr = threshold_scale * sigma * np.sqrt(2 * np.log(T + 1))

    # 对所有细节层做阈值处理（不对近似层处理）
    coeffs_thresh = [coeffs[0]]  # 保留近似系数不变
    for c in coeffs[1:]:
        if threshold_mode == 'soft':
            # 软阈值：收缩到0
            c_thresh = pywt.threshold(c, thr, mode='soft')
        else:
            # 硬阈值：直接置0
            c_thresh = pywt.threshold(c, thr, mode='hard')
        coeffs_thresh.append(c_thresh)

    # 重构
    x_rec = pywt.waverec(coeffs_thresh, wavelet)

    # 长度对齐（pywt 重构可能多1个点）
    return x_rec[:T].astype(np.float32)


def wavelet_denoise(wave: np.ndarray,
                    wavelet: str = 'db4',
                    level: int = None,
                    threshold_mode: str = 'soft',
                    threshold_scale: float = 1.0) -> np.ndarray:
    """
    三分量波形小波降噪

    wave : [3, T]
    return : [3, T]
    """
    out = np.zeros_like(wave)
    for ch in range(wave.shape[0]):
        out[ch] = wavelet_denoise_channel(
            wave[ch],
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
            threshold_scale=threshold_scale,
        )

    # 归一化
    m = np.abs(out).max()
    if m > 1e-10:
        out = out / m

    return out.astype(np.float32)


# ============================================================
#  单元测试
# ============================================================
if __name__ == "__main__":
    import time

    fs = 100.0
    T = 6000
    wave = np.random.randn(3, T).astype(np.float32)

    # BPF
    t0 = time.time()
    bpf = butterworth_bandpass(wave, fs=fs)
    print(f"BPF     : {time.time() - t0:.3f}s  shape={bpf.shape}")

    # Wavelet
    t0 = time.time()
    wav = wavelet_denoise(wave)
    print(f"Wavelet : {time.time() - t0:.3f}s  shape={wav.shape}")

    print("✅ 传统方法测试通过")