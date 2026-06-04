"""
TaiChiDataset
- 每个样本固定 103 帧，SEQ_LEN=64 / STRIDE=8 → 每样本约 5 个滑动窗口
- 骨骼归一化（髋部中心化 + 躯干尺度缩放）
- 数据增强（时间扰动 / 高斯噪声 / 关节遮掩 / 左右镜像）
"""
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Tuple, Optional
from config import FEATURE_DIM_POS, FEATURE_DIM_VEL, FEATURE_DIM
from config import *


# MediaPipe Pose 33关节镜像对（左右互换）
FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6),
    (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
]

# 左右对称动作对（1-indexed），翻转时需同步换标签
FLIP_LABEL_SWAP = {
    7: 8,   8: 7,    # 左揽雀尾 ↔ 右揽雀尾
    16: 17, 17: 16,  # 左下势独立 ↔ 右下势独立
}

class TaiChiDataset(Dataset):
    def __init__(self,
                 keypoints_dir: str   = KEYPOINTS_DIR,
                 split: str           = 'train',
                 seq_len: int         = SEQ_LEN,
                 stride: int          = STRIDE,
                 train_ratio: float   = TRAIN_RATIO,
                 augment: bool        = True):

        self.seq_len  = seq_len
        self.stride   = stride
        self.augment  = augment and (split == 'train')
        self.sequences: List[Tuple[np.ndarray, int]] = []

        pattern = re.compile(r'^(\d{2})_(\d{2})\.npy$')

        # 按 action_id 分组，保持样本顺序后划分 train/val
        action_groups: dict = defaultdict(list)
        for fname in sorted(os.listdir(keypoints_dir)):
            m = pattern.match(fname)
            if m:
                action_groups[int(m.group(1))].append(
                    (fname, int(m.group(2)))
                )

        total_files = 0
        for action_id, files in action_groups.items():
            files.sort(key=lambda x: x[1])  # 按 sample_id 排序
            n_train  = max(1, int(len(files) * train_ratio))
            selected = files[:n_train] if split == 'train' else files[n_train:]
            total_files += len(selected)

            for fname, _ in selected:
                kps = np.load(os.path.join(keypoints_dir, fname))  # (103, 99)
                kps = self._normalize(kps)

                # 固定103帧时：(103-64)//8 + 1 = 5 个窗口/样本
                T = len(kps)
                if T < seq_len:
                    # 理论上不会触发（固定103帧），保留为兜底逻辑
                    pad = np.tile(kps[-1:], (seq_len - T, 1))
                    kps = np.concatenate([kps, pad], axis=0)
                    T   = len(kps)

                for start in range(0, T - seq_len + 1, stride):
                    self.sequences.append(
                        (kps[start: start + seq_len].copy(),
                         action_id - 1)  # 0-indexed label
                    )

        wins_per_file = (FRAMES_PER_SAMPLE - seq_len) // stride + 1
        print(f"[{split:5s}] {len(self.sequences):5d} sequences  "
              f"({total_files} files × ~{wins_per_file} windows/file)")

    # ── 归一化 ──────────────────────────────────────────────────────
    def _normalize(self, kps: np.ndarray) -> np.ndarray:
        T      = len(kps)
        kps_3d = kps.reshape(T, 33, 3)

        hip_center      = (kps_3d[:, 23] + kps_3d[:, 24]) / 2.0
        shoulder_center = (kps_3d[:, 11] + kps_3d[:, 12]) / 2.0
        torso_len = np.linalg.norm(
            shoulder_center - hip_center, axis=1, keepdims=True
        )
        torso_len = np.clip(torso_len, 1e-6, None)

        kps_norm = (kps_3d - hip_center[:, np.newaxis, :]) \
                / torso_len[:, np.newaxis, :]
        pos = kps_norm.reshape(T, FEATURE_DIM_POS)    # (T, 99)

        # ── 速度特征：帧间位置差分 ──────────────────────────────
        vel      = np.zeros_like(pos)
        vel[1:]  = pos[1:] - pos[:-1]
        vel[0]   = vel[1]   # 首帧用第2帧速度填充

        return np.concatenate([pos, vel], axis=-1)    # (T, 198)

    def _augment(self, seq: np.ndarray, label_0idx: int
                 ) -> Tuple[np.ndarray, int]:
        """
        返回 (augmented_seq, possibly_swapped_label)
        label_0idx: 0-indexed 标签
        """
        # 1. 时间速度扰动
        if np.random.rand() < 0.5:
            T       = len(seq)
            scale   = np.random.uniform(0.8, 1.25)
            n_new   = max(int(T * scale), T // 2)
            src_idx = np.round(np.linspace(0, T-1, n_new)).astype(int)
            seq     = seq[src_idx]
            dst_idx = np.round(np.linspace(0, len(seq)-1, T)).astype(int)
            seq     = seq[dst_idx]

        # 2. 高斯噪声
        if np.random.rand() < 0.5:
            seq = seq + np.random.normal(0, 0.025, seq.shape).astype(np.float32)

        # 3. 随机关节遮掩
        if np.random.rand() < 0.3:
            n_mask = np.random.randint(1, 4)
            joints = np.random.choice(33, size=n_mask, replace=False)
            for j in joints:
                seq[:, j*3: j*3+3] = 0.0

        # 4. ✅ 左右镜像——同步修改标签
        if np.random.rand() < 0.5:
            seq = self._horizontal_flip(seq)
            label_1idx     = label_0idx + 1
            new_label_1idx = FLIP_LABEL_SWAP.get(label_1idx, label_1idx)
            label_0idx     = new_label_1idx - 1

        return seq, label_0idx

    def _horizontal_flip(self, seq):
        T = len(seq)
        
        # 拆分 pos (前99) 和 vel (后99)
        pos = seq[:, :99].reshape(T, 33, 3).copy()
        vel = seq[:, 99:].reshape(T, 33, 3).copy()

        # 翻转 x 坐标
        pos[:, :, 0] *= -1
        vel[:, :, 0] *= -1   # velocity 是差值，直接取反即可

        # 交换左右关键点
        SWAP_PAIRS = [
            (11, 12), (13, 14), (15, 16),   # 肩、肘、腕
            (17, 18), (19, 20), (21, 22),   # 手指
            (23, 24), (25, 26), (27, 28),   # 髋、膝、踝
            (29, 30), (31, 32),             # 脚跟、脚趾
        ]
        for l, r in SWAP_PAIRS:
            pos[:, [l, r]] = pos[:, [r, l]]
            vel[:, [l, r]] = vel[:, [r, l]]

        # 重新拼回 (T, 198)
        return np.concatenate([pos.reshape(T, 99), vel.reshape(T, 99)], axis=-1)

    # ── Dataset 接口 ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq, label = self.sequences[idx]
        seq = seq.copy()
        if self.augment:
            seq, label = self._augment(seq, label)   # ← 接收可能变化的label
        return torch.FloatTensor(seq), torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size: int = BATCH_SIZE):
    train_ds = TaiChiDataset(split='train', augment=True)
    val_ds   = TaiChiDataset(split='val',   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader