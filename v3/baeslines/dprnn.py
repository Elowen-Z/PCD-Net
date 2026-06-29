# v3/baselines/dprnn.py
"""
DPRNN: Dual-Path RNN for seismic denoising
参考: Luo et al. (2020) "Dual-path RNN: efficient long sequence
      modeling for time-domain single-channel speech separation"

改造为地震三分量降噪：
  输入: x [B, 3, T]
  输出: clean [B, 3, T]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRNNBlock(nn.Module):
    """
    双路径 RNN 块：
      行内 RNN（chunk 内时序建模）+ 行间 RNN（全局依赖）
    """
    def __init__(self, feat_dim: int, hidden_dim: int,
                 chunk_size: int = 100, dropout: float = 0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.feat_dim   = feat_dim

        # 行内（局部）
        self.intra_rnn  = nn.LSTM(
            feat_dim, hidden_dim, batch_first=True,
            bidirectional=True
        )
        self.intra_proj = nn.Linear(hidden_dim * 2, feat_dim)
        self.intra_norm = nn.LayerNorm(feat_dim)

        # 行间（全局）
        self.inter_rnn  = nn.LSTM(
            feat_dim, hidden_dim, batch_first=True,
            bidirectional=True
        )
        self.inter_proj = nn.Linear(hidden_dim * 2, feat_dim)
        self.inter_norm = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, feat_dim, T]"""
        B, C, T = x.shape
        S = self.chunk_size
        # padding
        pad = (S - T % S) % S
        if pad > 0:
            x = F.pad(x, (0, pad))
        T_pad = x.shape[-1]
        n_chunks = T_pad // S

        # ── 行内 RNN ─────────────────────────────────
        # [B, C, n_chunks, S] → [B*n_chunks, S, C]
        xc = x.view(B, C, n_chunks, S)
        xc = xc.permute(0, 2, 3, 1).contiguous()   # [B, n, S, C]
        xc = xc.view(B * n_chunks, S, C)
        out, _ = self.intra_rnn(xc)                 # [B*n, S, 2H]
        out = self.dropout(self.intra_proj(out))     # [B*n, S, C]
        out = out.view(B, n_chunks, S, C)
        out = self.intra_norm(out)
        # 残差
        xc  = xc.view(B, n_chunks, S, C)
        xc  = self.intra_norm(xc + out)             # [B, n, S, C]

        # ── 行间 RNN ─────────────────────────────────
        # [B, n, S, C] → [B*S, n, C]
        xi = xc.permute(0, 2, 1, 3).contiguous()   # [B, S, n, C]
        xi = xi.view(B * S, n_chunks, C)
        out2, _ = self.inter_rnn(xi)                # [B*S, n, 2H]
        out2 = self.dropout(self.inter_proj(out2))  # [B*S, n, C]
        out2 = out2.view(B, S, n_chunks, C)
        out2 = out2.permute(0, 2, 1, 3)            # [B, n, S, C]
        out2 = self.inter_norm(xc + out2)

        # 还原 [B, C, T_pad]
        out2 = out2.permute(0, 3, 1, 2).contiguous()  # [B, C, n, S]
        out2 = out2.view(B, C, T_pad)

        # 去掉 padding
        if pad > 0:
            out2 = out2[..., :T]
        return out2

class DPRNN(nn.Module):
    """
    编码 → N×DPRNN块 → 解码
    """
    def __init__(
        self,
        in_ch:      int = 3,
        feat_dim:   int = 64,
        hidden_dim: int = 128,
        n_blocks:   int = 4,
        chunk_size: int = 100,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # 输入编码（升维）
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, feat_dim, kernel_size=16,
                      stride=8, padding=4),
            nn.PReLU(),
        )

        # DPRNN 堆叠
        self.dprnn_blocks = nn.ModuleList([
            DPRNNBlock(feat_dim, hidden_dim, chunk_size, dropout)
            for _ in range(n_blocks)
        ])

        # 输出解码（降维 + 还原长度）
        self.decoder = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(feat_dim, in_ch * 8, kernel_size=1),  # 8 = stride
        )

        self.out_act = nn.Tanh()
        self.in_ch   = in_ch

    def forward(self, x, z_cond=None):
        """
        z_cond: 忽略（统一接口）
        返回: (clean [B,3,T], None, None)
        """
        T_orig = x.shape[-1]

        enc = self.encoder(x)             # [B, feat_dim, T//8]

        h = enc
        for block in self.dprnn_blocks:
            h = block(h)                  # [B, feat_dim, T//8]

        dec = self.decoder(h)             # [B, in_ch*8, T//8]

        # 像素shuffle还原时间维度
        B, C8, L = dec.shape
        dec = dec.view(B, self.in_ch, 8, L)
        dec = dec.permute(0, 1, 3, 2).contiguous()
        dec = dec.view(B, self.in_ch, L * 8)

        # 对齐到原始长度
        if dec.shape[-1] != T_orig:
            dec = F.interpolate(dec, size=T_orig,
                                mode='linear', align_corners=False)

        out = self.out_act(dec)
        return out, None, None