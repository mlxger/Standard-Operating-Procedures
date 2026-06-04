"""
TaiChiTransformer
────────────────────────────────────────────────────────
输入:  (B, seq_len=64, feature_dim=99)
输出:  (B, 24)
参数量: ~350K  显存: <1GB (batch=64)
────────────────────────────────────────────────────────
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class DepthwiseTCN(nn.Module):
    """
    多尺度空洞深度卷积
    dilation = 1/2/4 覆盖 1帧/2帧/4帧邻域，提取局部运动曲率
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            self._dw_conv(d_model, dilation=d) for d in [1, 2, 4]
        ])
        self.merge = nn.Linear(d_model * 3, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    @staticmethod
    def _dw_conv(d_model: int, dilation: int) -> nn.Sequential:
        pad = 2 * dilation          # kernel=3, 因果填充
        return nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3,
                      padding=pad, dilation=dilation,
                      groups=d_model),           # depthwise
            nn.Conv1d(d_model, d_model, 1),      # pointwise
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T   = x.size(1)
        xt  = x.transpose(1, 2)                          # (B, d, T)
        outs = [c(xt)[:, :, :T] for c in self.convs]    # trim因果padding
        out  = torch.cat(outs, dim=1).transpose(1, 2)   # (B, T, 3d)
        return self.norm(x + self.drop(self.merge(out))) # residual


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TaiChiEncoderBlock(nn.Module):
    """Pre-Norm Transformer Block（训练更稳定）"""

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm Self-Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # Pre-Norm Feed-Forward
        x = x + self.ff(self.norm2(x))
        return x


class TaiChiTransformer(nn.Module):
    """
    ┌─────────────────────────────────────────────────────┐
    │                TaiChiTransformer                    │
    │                                                     │
    │  Input (B,64,99)                                    │
    │      ↓ Linear Projection                            │
    │  (B,64,128)                                         │
    │      ↓ + CLS token → (B,65,128)                    │
    │      ↓ Positional Encoding                          │
    │  TransformerBlock × 4                               │
    │      ↓                                              │
    │  CLS ‖ AvgPool ‖ MaxPool → (B, 384)                │
    │      ↓ MLP Head                                     │
    │  Logits (B, 24)                                     │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        feature_dim:     int = FEATURE_DIM,
        d_model:         int = D_MODEL,
        nhead:           int = NHEAD,
        num_layers:      int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout:         float = DROPOUT,
        num_classes:     int = NUM_CLASSES,
    ):
        super().__init__()
        self.d_model = d_model

        # ── 1. 输入映射 ──────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.tcn = DepthwiseTCN(d_model, dropout) 

        # ── 2. 可学习 [CLS] token ────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. 位置编码 ──────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # ── 4. Transformer Encoder ───────────────────────────────
        self.blocks = nn.ModuleList([
            TaiChiEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── 5. 多尺度时序池化 ────────────────────────────────────
        # CLS(d) + AvgPool(d) + MaxPool(d) = 3d
        head_in = d_model * 3

        # ── 6. 分类头 ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 99)
        returns: (B, 24) logits
        """
        B, T, _ = x.shape

        # 输入映射
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.tcn(x)           # (B, T, d_model)
        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)         # (B, T+1, d_model)

        # 位置编码
        x = self.pos_enc(x)

        # Transformer 编码
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # (B, T+1, d_model)

        # 多尺度特征提取
        cls_out = x[:, 0]                                # (B, d_model)
        seq_out = x[:, 1:].transpose(1, 2)              # (B, d_model, T)
        avg_out = F.adaptive_avg_pool1d(seq_out, 1).squeeze(-1)  # (B, d_model)
        max_out = F.adaptive_max_pool1d(seq_out, 1).squeeze(-1)  # (B, d_model)

        feat   = torch.cat([cls_out, avg_out, max_out], dim=-1)  # (B, 3d)
        logits = self.head(feat)                                   # (B, 24)

        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """返回 (pred_class, confidence, all_probs)"""
        self.eval()
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=-1)
        pred   = probs.argmax(dim=-1)
        conf   = probs.max(dim=-1).values
        return pred, conf, probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TaiChiTransformer()
    print(f"参数量: {count_parameters(model):,}")    # ~350K

    dummy = torch.randn(4, SEQ_LEN, FEATURE_DIM)
    out   = model(dummy)
    print(f"输入: {dummy.shape}  输出: {out.shape}")  # (4, 24)