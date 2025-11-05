import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import C2AE
from evt_utils import calculate_threshold
import os

parser = argparse.ArgumentParser(description='C2AE Open-set Recognition')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs-stage1', type=int, default=20, metavar='N',
                    help='number of epochs for stage 1 (default: 20)')
parser.add_argument('--epochs-stage2', type=int, default=100, metavar='N',
                    help='number of epochs for stage 2 (default: 100)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.9, metavar='ALPHA',
                    help='weight for match loss (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-known-classes', type=int, default=6, metavar='K',
                    help='number of known classes (default: 6)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging')
parser.add_argument('--data-dir', type=str, default='../data/', metavar='DIR',
                    help='directory for dataset')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# 划分开闭集
known_classes = list(range(args.num_known_classes))
train_loader, test_known_loader, test_unknown_loader = get_dataloaders(
    known_classes, args.batch_size, args.data_dir
)

print(f"Known classes: {known_classes}")
print(f"Unknown classes: {[i for i in range(10) if i not in known_classes]}")
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Test samples (known): {len(test_known_loader.dataset)}")
print(f"Test samples (unknown): {len(test_unknown_loader.dataset)}")

# 导入模型
model = C2AE(latent_dim=128, num_classes=args.num_known_classes)
if args.cuda:
    model.cuda()
print(model)

######################## stage1: 闭集分类训练与测试 ########################
print("\n" + "="*60)
print("Stage 1: Closed-set Classification Training")
print("="*60)

criterion_stage1 = nn.CrossEntropyLoss()
optimizer_stage1 = torch.optim.Adam(
    list(model.encoder.parameters()) + list(model.classifier.parameters()),
    lr=args.lr
)

def train_stage1(epoch):
    model.train()
    model.decoder.eval()
    
    for i, (images, labels) in enumerate(train_loader):
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        
        optimizer_stage1.zero_grad()
        outputs = model(images, mode='stage1')
        loss = criterion_stage1(outputs, labels)
        loss.backward()
        optimizer_stage1.step()
        
        if (i + 1) % args.log_interval == 0:
            print(f'Epoch [{epoch+1}/{args.epochs_stage1}], '
                    f'Iter [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

def test_stage1():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_known_loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images, mode='stage1')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Stage 1 Test Accuracy: {accuracy:.2f}%')
    return accuracy

for epoch in range(args.epochs_stage1):
    train_stage1(epoch)
    if (epoch + 1) % 5 == 0:
        test_stage1()

######################## stage2: 开集训练与测试 ########################
print("\n" + "="*60)
print("Stage 2: Open-set Training with Modified Non-Match Loss")
print("="*60)

# 锁定stage1的编码器和分类器参数
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = False

optimizer_stage2 = torch.optim.Adam(
    list(model.film.parameters()) + list(model.decoder.parameters()),
    lr=args.lr
)

def train_stage2(epoch):
    model.train()
    model.encoder.eval()
    model.classifier.eval()
    
    alpha = args.alpha
    
    for i, (images, labels) in enumerate(train_loader):
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        
        optimizer_stage2.zero_grad()
        
        # 同一批图像，两种条件
        x_recon_match, x_recon_nonmatch = model(images, labels, mode='stage2')
        
        # 匹配损失: L_m^r = ||X_i - X̃_i^m||_1
        loss_match = torch.mean(torch.sum(torch.abs(images - x_recon_match), dim=[1,2,3]))
        
        # 不匹配损失: L_{nm}^r = ||X_i - X̃_i^{nm}||_1
        loss_nonmatch = torch.mean(torch.sum(torch.abs(images - x_recon_nonmatch), dim=[1,2,3]))
        
        # # 总损失：让匹配重构好，非匹配重构差
        loss = alpha * loss_match - (1 - alpha) * loss_nonmatch
        
        loss.backward()
        optimizer_stage2.step()

        if (i + 1) % args.log_interval == 0:
            print(f'Epoch [{epoch+1}/{args.epochs_stage2}] (α={alpha:.2f}), '
                    f'Iter [{i+1}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f} '
                    f'(Match: {loss_match.item():.4f}, NonMatch: {loss_nonmatch.item():.4f})')

def compute_reconstruction_errors(loader):
    model.eval() 
    match_errors = []
    nonmatch_errors = []
    
    with torch.no_grad():
        for images, labels in loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            
            z = model.encoder(images)
            
            # 匹配重构的重建误差
            condition_match = model.create_condition_vector(labels)
            z_match = model.film(z, condition_match)
            x_recon_match = model.decoder(z_match)
            error_match = torch.sum(torch.abs(images - x_recon_match), dim=[1,2,3])
            match_errors.extend(error_match.cpu().numpy())
            
            # 非匹配重构
            labels_nm = torch.randint(0, model.num_classes, labels.size(), device=labels.device)
            mask = (labels_nm == labels)
            labels_nm[mask] = (labels_nm[mask] + 1) % model.num_classes
            condition_nonmatch = model.create_condition_vector(labels_nm)

            z_nm = model.film(z, condition_nonmatch)
            x_recon_nm = model.decoder(z_nm)
            error_nm = torch.sum(torch.abs(images - x_recon_nm), dim=[1,2,3])
            nonmatch_errors.extend(error_nm.cpu().numpy())
    
    return np.array(match_errors), np.array(nonmatch_errors)

output_dir = '../reconstruction_hist'
os.makedirs(output_dir, exist_ok=True)

for epoch in range(args.epochs_stage2):
    train_stage2(epoch)
    # 每10个epoch绘制一次重建误差直方图
    # if (epoch + 1) % 10 == 0:
    #     S_m_temp, S_nm_temp = compute_reconstruction_errors(train_loader)
    #     plt.figure(figsize=(12, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.hist(S_m_temp, bins=50, alpha=0.5, color='yellow', label='Match')
    #     plt.hist(S_nm_temp, bins=50, alpha=0.5, color='blue', label='Non-Match')
    #     plt.legend()
    #     plt.title(f'Epoch {epoch+1}')
    #     plt.xlabel('Reconstruction Error')
    #     plt.ylabel('Normalized Histogram')
    #     save_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.png')
    #     plt.savefig(save_path)
    #     plt.close()

######################## 计算重建误差与阈值 ########################
print("\n" + "="*60)
print("Computing Reconstruction Errors with EVT Modeling")
print("="*60)

# 锁定Stage2训练过的FiLM模块和解码器参数
for param in model.film.parameters():
    param.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = False

S_m, S_nm = compute_reconstruction_errors(train_loader)
print(f"Match errors S_m: size={len(S_m)}, mean={S_m.mean():.4f}, std={S_m.std():.4f}")
print(f"Non-match errors S_nm: size={len(S_nm)}, mean={S_nm.mean():.4f}, std={S_nm.std():.4f}")

threshold = calculate_threshold(S_m, S_nm, p_u=0.5)
print(f"Optimal Threshold τ*: {threshold:.4f}")

######################## stage3: 开集测试 ########################
print("\n" + "="*60)
print("Stage 3: Open-set Testing (k-inference)")
print("="*60)

def test_openset():
    model.eval()
    all_labels = []
    all_recon_errors = []

    with torch.no_grad():
        # 已知类（闭集）
        for images, _ in test_known_loader:
            if args.cuda:
                images = images.cuda()
            _, min_recon_error = model(images, mode='stage3')
            all_labels.extend([1] * images.size(0))
            all_recon_errors.extend(min_recon_error.cpu().numpy())

        # 未知类（开集）
        for images, _ in test_unknown_loader:
            if args.cuda:
                images = images.cuda()
            _, min_recon_error = model(images, mode='stage3')
            all_labels.extend([0] * images.size(0))
            all_recon_errors.extend(min_recon_error.cpu().numpy())

    all_labels = np.array(all_labels)
    all_recon_errors = np.array(all_recon_errors)

    # AUROC
    auroc = roc_auc_score(all_labels, -all_recon_errors)

    # F-measure
    predictions = (all_recon_errors <= threshold).astype(int)
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f_measure = 2 * precision * recall / (precision + recall + 1e-10)

    print(f"\nAUROC (Open-set Identification): {auroc:.4f}")
    print(f"F-measure: {f_measure:.4f}")
    return auroc, f_measure

auroc, f_measure = test_openset()

torch.save({
    'encoder': model.encoder.state_dict(),
    'classifier': model.classifier.state_dict(),
    'film': model.film.state_dict(),
    'decoder': model.decoder.state_dict(),
    'threshold': threshold,
    'auroc': auroc,
    'f_measure': f_measure,
    'args': args
}, '../c2ae_model.pkl')