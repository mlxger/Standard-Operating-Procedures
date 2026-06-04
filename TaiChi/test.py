from dataset import get_dataloaders

if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders(batch_size=4)
    seq, label = next(iter(train_loader))
    print(seq.shape)   # 期望: torch.Size([4, 64, 198])
    print(label.shape) # 期望: torch.Size([4])