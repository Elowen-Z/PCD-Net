# v3/baselines/deep_denoiser.py
"""
DeepDenoiser: 地震波形降噪经典 U-Net
参考: Zhu et al. (2019) "Seismic Signal Denoising and Decomposition
      Using Deep Neural Networks"

输入: x [B, 3, T]
输出: clean [B, 3, T]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      stride=stride, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class DeepDenoiser(nn.Module):
    """
    5层 U-Net，无条件调制，无 Attention
    最接近原版 DeepDenoiser 结构
    """
    def __init__(self, in_ch: int = 3):
        super().__init__()

        # ── Encoder ──────────────────────────────────
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, 32,  kernel=3, stride=2),
            ConvBNReLU(32,    32,  kernel=3, stride=1),
        )
        self.enc2 = nn.Sequential(
            ConvBNReLU(32,  64,  kernel=3, stride=2),
            ConvBNReLU(64,  64,  kernel=3, stride=1),
        )
        self.enc3 = nn.Sequential(
            ConvBNReLU(64,  128, kernel=3, stride=2),
            ConvBNReLU(128, 128, kernel=3, stride=1),
        )
        self.enc4 = nn.Sequential(
            ConvBNReLU(128, 256, kernel=3, stride=2),
            ConvBNReLU(256, 256, kernel=3, stride=1),
        )

        # ── Bottleneck ───────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBNReLU(256, 512, kernel=3, stride=2),
            ConvBNReLU(512, 512, kernel=3, stride=1),
            ConvBNReLU(512, 256, kernel=3, stride=1),
        )

        # ── Decoder ──────────────────────────────────
        self.dec4 = nn.Sequential(
            ConvBNReLU(256 + 256, 256, kernel=3, stride=1),
            ConvBNReLU(256,       128, kernel=3, stride=1),
        )
        self.dec3 = nn.Sequential(
            ConvBNReLU(128 + 128, 128, kernel=3, stride=1),
            ConvBNReLU(128,       64,  kernel=3, stride=1),
        )
        self.dec2 = nn.Sequential(
            ConvBNReLU(64 + 64, 64, kernel=3, stride=1),
            ConvBNReLU(64,      32, kernel=3, stride=1),
        )
        self.dec1 = nn.Sequential(
            ConvBNReLU(32 + 32, 32, kernel=3, stride=1),
            ConvBNReLU(32,      32, kernel=3, stride=1),
        )

        # ── 输出 ─────────────────────────────────────
        # 线性输出 (去掉 Tanh, 避免幅度饱和削顶); 采用残差学习, 网络只需预测噪声增量
        self.out_conv = nn.Conv1d(32, in_ch, kernel_size=1)

    @staticmethod
    def _align(x, target):
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(x, size=target.shape[-1],
                              mode='linear', align_corners=False)
        return x

    def forward(self, x, z_cond=None):
        """
        z_cond: 忽略（接口统一，兼容评估脚本）
        返回: (clean [B,3,T], None, None)  ← 统一接口
        """
        e1 = self.enc1(x)                                      # T/2
        e2 = self.enc2(e1)                                     # T/4
        e3 = self.enc3(e2)                                     # T/8
        e4 = self.enc4(e3)                                     # T/16

        bn = self.bottleneck(e4)                               # T/32

        d4 = F.interpolate(bn, scale_factor=2,
                           mode='linear', align_corners=False)
        d4 = self._align(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, scale_factor=2,
                           mode='linear', align_corners=False)
        d3 = self._align(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2,
                           mode='linear', align_corners=False)
        d2 = self._align(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2,
                           mode='linear', align_corners=False)
        d1 = self._align(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = F.interpolate(d1, scale_factor=2,
                           mode='linear', align_corners=False)
        d0 = self._align(d0, x)

        # 残差学习: 输出 = 输入 - 预测噪声 (线性, 不再经 Tanh)
        out = x + self.out_conv(d0)
        return out, None, None