import os
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Dict, Optional
import yaml


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class ActionTransformer(nn.Module):
    def __init__(self, input_dim=126, num_classes=10, seq_len=30,
                 d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 10)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.classifier(x.mean(dim=1))


class ActionRecognizer:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # ✅ 修复：添加 encoding='utf-8'
        with open(config_path, 'r', encoding='utf-8') as f:
            all_cfg = yaml.safe_load(f)

        cfg = all_cfg['action_model']
        self.seq_len     = cfg['seq_len']
        self.device      = cfg['device'] if torch.cuda.is_available() else 'cpu'
        self.num_classes = cfg['num_classes']
        self.buffer      = deque(maxlen=self.seq_len)

        label_cfg    = all_cfg.get('action_labels', {})
        self.labels  = {int(k): v for k, v in label_cfg.items()}

        self.model = ActionTransformer(
            input_dim=cfg['input_dim'],
            num_classes=cfg['num_classes'],
            seq_len=cfg['seq_len'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers']
        ).to(self.device)

        model_path = "models/action_model.pth"
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"[ActionRecognizer] ✅ 加载模型: {model_path}")
        else:
            print("[ActionRecognizer] ⚠️  未找到预训练权重，使用随机初始化")
            print("   请先运行: python tools/train_action.py")

        self.model.eval()
        self._warmup()

    def _warmup(self):
        dummy = torch.zeros(1, self.seq_len, 126).to(self.device)
        with torch.no_grad():
            self.model(dummy)

    def update(self, feature_vector: np.ndarray) -> Optional[Dict]:
        self.buffer.append(feature_vector.astype(np.float32))
        if len(self.buffer) < self.seq_len:
            return None
        return self._inference()

    def _inference(self) -> Dict:
        sequence = np.array(list(self.buffer))
        with torch.no_grad():
            x      = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1)[0]
            pred   = probs.argmax().item()
            conf   = probs[pred].item()
        return {
            'action_id':   pred,
            'action_name': self.labels.get(pred, f'Action_{pred}'),
            'confidence':  conf,
            'all_probs':   probs.cpu().numpy()
        }

    def clear_buffer(self):
        self.buffer.clear()