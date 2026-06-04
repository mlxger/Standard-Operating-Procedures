"""
训练脚本
- AdamW + CosineAnnealing + WarmRestart
- Label Smoothing
- AMP 混合精度
- 自动保存最优模型 + 训练曲线
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from config import *
from model import TaiChiTransformer, count_parameters
from dataset import get_dataloaders


# ── Warmup + Cosine 调度器 ─────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor   = 0.5 * (1 + np.cos(np.pi * progress))
            factor   = max(factor, self.min_lr / self.base_lrs[0])

        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base_lr * factor


# ── 单轮训练 ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0., 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for seqs, labels in pbar:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(seqs)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs          = seqs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(-1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


# ── 验证 ───────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []

    for seqs, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=90):
        seqs, labels = seqs.to(device), labels.to(device)
        logits       = model(seqs)
        loss         = criterion(logits, labels)

        bs          = seqs.size(0)
        total_loss += loss.item() * bs
        preds       = logits.argmax(-1)
        correct    += (preds == labels).sum().item()
        total      += bs

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ── 绘图 ───────────────────────────────────────────────────────
def save_training_curve(history, log_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes,
                               [('train_loss','val_loss'),
                                ('train_acc','val_acc')],
                               ['Loss', 'Accuracy']):
        ax.plot(history[key[0]], label='Train')
        ax.plot(history[key[1]], label='Val')
        ax.set_title(title); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curve.png'), dpi=100)
    plt.close()


def save_confusion_matrix(preds, labels, log_dir):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm     = confusion_matrix(labels, preds)
    ticks  = [f"{i+1:02d}" for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ticks, yticklabels=ticks, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - TaiChi 24')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'confusion_matrix_best.png'), dpi=100)
    plt.close()


# ── 主训练函数 ─────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 数据
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    # 模型
    model = TaiChiTransformer().to(device)
    print(f"模型参数量: {count_parameters(model):,}\n")

    # 损失（Label Smoothing 防过拟合）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # 调度器
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=10,
                                      total_epochs=EPOCHS)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    history      = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc']}
    best_val_acc = 0.0

    print("=" * 60)
    print("  开始训练 TaiChiTransformer")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        scheduler.step(epoch - 1)

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader,
                                      optimizer, criterion, device, scaler)
        vl_loss, vl_acc, preds, labels = val_epoch(model, val_loader,
                                                    criterion, device)
        elapsed = time.time() - t0

        for k, v in zip(['train_loss','train_acc','val_loss','val_acc'],
                        [tr_loss, tr_acc, vl_loss, vl_acc]):
            history[k].append(v)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:3d}/{EPOCHS}] "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f} | "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        # 保存最优
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
            }
            torch.save(ckpt,
                       os.path.join(MODEL_SAVE_DIR,
                                    'taichi_transformer_best.pth'))
            save_confusion_matrix(preds, labels, LOG_DIR)
            print(f"  ✓ 保存最优模型 val_acc={vl_acc:.4f}")

        # 每20轮保存训练曲线
        if epoch % 20 == 0:
            save_training_curve(history, LOG_DIR)

    save_training_curve(history, LOG_DIR)
    print(f"\n训练完成！最优 val_acc = {best_val_acc:.4f}")

    # 最终分类报告
    print("\n── 分类报告 (最优模型) ─────────────────")
    best_ckpt = torch.load(
        os.path.join(MODEL_SAVE_DIR, 'taichi_transformer_best.pth'),
        map_location=device
    )
    model.load_state_dict(best_ckpt['model_state_dict'])
    _, _, preds, labels = val_epoch(model, val_loader, criterion, device)
    names = [ACTION_NAMES[i+1] for i in range(NUM_CLASSES)]
    print(classification_report(labels, preds, target_names=names, digits=3))


if __name__ == "__main__":
    train()