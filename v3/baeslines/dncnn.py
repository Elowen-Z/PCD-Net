# v3/baeslines/dncnn.py
"""
DnCNN-1D: 残差学习去噪 (Zhang et al., 2017 IEEE TIP)
为 1D 三分量地震信号改写：输入 [B,3,T]，输出 [B,3,T]。
网络预测残差噪声 r，干净输出 clean = x - r。
"""
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    经典 DnCNN：1*Conv-ReLU + (D-2)*Conv-BN-ReLU + 1*Conv
    默认 depth=17, base_ch=64（与原文一致）
    """

    def __init__(self, in_ch: int = 3, depth: int = 17, base_ch: int = 64,
                 kernel: int = 7):
        super().__init__()
        pad = kernel // 2

        layers = [nn.Conv1d(in_ch, base_ch, kernel, padding=pad),
                  nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv1d(base_ch, base_ch, kernel, padding=pad, bias=False),
                nn.BatchNorm1d(base_ch),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv1d(base_ch, in_ch, kernel, padding=pad)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        return x - residual
