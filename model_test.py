import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from model import CeiT

def get_dataloaders(batch_size=32):
    """获取 CIFAR-10 验证数据加载器"""
    val_transform = transforms.Compose([
        transforms.Resize(64),  # 与训练一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_data = CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return val_loader

def test(model, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n验证集/测试集准确率: {acc:.4f}")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建 CeiT 模型实例（与训练时一致）
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

    # 加载训练好的最佳模型
    model.load_state_dict(torch.load("best_model.pth"))

    # 测试
    test(model, get_dataloaders(batch_size=32))
