"""
运行方式: python tools/train_action.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import yaml
from src.action_recognizer import ActionTransformer


class KeypointDataset(Dataset):
    def __init__(self, data_dir: str = "data/keypoints"):
        self.samples = []
        self.labels  = []
        
        for class_dir in sorted(Path(data_dir).iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for npy_file in class_dir.glob("*.npy"):
                seq = np.load(str(npy_file)).astype(np.float32)
                self.samples.append(seq)
                self.labels.append(label)
        
        print(f"[Dataset] 共 {len(self.samples)} 个样本, "
              f"{len(set(self.labels))} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.samples[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


def train():
    # 配置
    EPOCHS      = 100
    BATCH_SIZE  = 32
    LR          = 3e-4
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH   = 'models/action_model.pth'
    
    with open('configs/model_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['action_model']
    
    # 数据
    dataset = KeypointDataset()
    assert len(dataset) > 0, "请先运行 tools/data_collector.py 采集数据"
    
    n_val   = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
    
    # 模型
    model = ActionTransformer(
        input_dim=cfg['input_dim'],
        num_classes=cfg['num_classes'],
        seq_len=cfg['seq_len'],
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers']
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(EPOCHS):
        # ── 训练 ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss    += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += y.size(0)
        
        # ── 验证 ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total   += y.size(0)
        
        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'val_acc':           val_acc,
                'config':            cfg
            }, SAVE_PATH)
            print(f"  ✅ 保存最优模型 val_acc={val_acc:.1f}%")
    
    print(f"\n训练完成! 最优验证准确率: {best_val_acc:.1f}%")
    print(f"模型已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    train()