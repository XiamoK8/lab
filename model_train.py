import copy
import time
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import CeiT
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_dataloaders(batch_size=64):
    """获取CIFAR10数据加载器，添加数据增强"""
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(64, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_data = CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = Data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train(model, train_loader, val_loader, epochs, learning_rate=3e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []

    since = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)

        train_loss, train_correct, train_num = 0.0, 0.0, 0
        val_loss, val_correct, val_num = 0.0, 0.0, 0

        # 训练阶段
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            train_num += imgs.size(0)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                val_num += imgs.size(0)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        train_loss_avg = train_loss / train_num
        val_loss_avg = val_loss / val_num
        train_acc = train_correct / train_num
        val_acc = val_correct / val_num

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "best_model.pth")
            print(f"✓ 最佳模型已保存 (Val Acc: {best_acc:.4f})")

        time_used = time.time() - since
        print(f"累计耗时: {int(time_used//60)}m {int(time_used%60)}s\n")

    train_result = pd.DataFrame({
        "epoch": np.arange(1, epochs+1),
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates
    })

    return train_result

def plot_curve(train_result):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(train_result["epoch"], train_result.train_losses, "ro-", linewidth=2, label="train loss")
    axes[0, 0].plot(train_result["epoch"], train_result.val_losses, "bs-", linewidth=2, label="val loss")
    axes[0, 0].set_xlabel("epoch", fontsize=12)
    axes[0, 0].set_ylabel("loss", fontsize=12)
    axes[0, 0].set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_result["epoch"], train_result.train_accuracies, "ro-", linewidth=2, label="train acc")
    axes[0, 1].plot(train_result["epoch"], train_result.val_accuracies, "bs-", linewidth=2, label="val acc")
    axes[0, 1].set_xlabel("epoch", fontsize=12)
    axes[0, 1].set_ylabel("accuracy", fontsize=12)
    axes[0, 1].set_title("Training and Validation Accuracy", fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(train_result["epoch"], train_result.learning_rates, "g-", linewidth=2)
    axes[1, 0].set_xlabel("epoch", fontsize=12)
    axes[1, 0].set_ylabel("learning rate", fontsize=12)
    axes[1, 0].set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    fig.delaxes(axes[1,1])

    plt.tight_layout()
    plt.savefig("result.png", dpi=300, bbox_inches="tight")
    print(f"\n训练曲线已保存至: result.png")
    plt.show()


if __name__ == '__main__':
    print("="*60)
    print("CeiT on CIFAR10 训练开始")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CeiT(
        img_size=64,
        patch_size=8,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.05,
        leff_local_size=3,
        leff_with_bn=True
    ).to(device)

    print("✓ 模型已创建\n")

    print("加载CIFAR10数据集...")
    train_loader, val_loader = get_dataloaders(batch_size=64)
    print(f"✓ 训练集大小: {len(train_loader.dataset)}")
    print(f"✓ 验证集大小: {len(val_loader.dataset)}\n")

    train_result = train(model, train_loader, val_loader, epochs=100, learning_rate=3e-4)
    plot_curve(train_result)

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
