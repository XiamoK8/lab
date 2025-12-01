import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import CIFAR10_c2ae
from metrics import evaluate
from evt_utils import calculate_threshold
from model import C2AE


def build_arg_parser():
    parser = argparse.ArgumentParser(description="C2AE Open-set Recognition")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--epochs-stage1", type=int, default=20, metavar="N",
                        help="number of epochs for stage 1 (default: 20)")
    parser.add_argument("--epochs-stage2", type=int, default=100, metavar="N",
                        help="number of epochs for stage 2 (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0003, metavar="LR",
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--alpha", type=float, default=0.9, metavar="ALPHA",
                        help="weight for match loss (default: 0.9)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="enables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--num-known-classes", type=int, default=6, metavar="K",
                        help="number of known classes (default: 6)")
    parser.add_argument("--log-interval", type=int, default=50, metavar="N",
                        help="how many batches to wait before logging")
    parser.add_argument("--data-dir", type=str, default="../cifar10", metavar="DIR",
                        help="directory for CIFAR-10 data root (expects subfolders train/ and test/)")
    return parser


def set_seed(seed, use_cuda):
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True


def train_closed_set(model, train_loader, optimizer_stage1, criterion_stage1, epoch, args):
    model.train()
    model.decoder.eval()

    for i, (images, labels) in enumerate(train_loader):
        if args.cuda:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer_stage1.zero_grad()
        outputs = model(images, mode="stage1")
        loss = criterion_stage1(outputs, labels)
        loss.backward()
        optimizer_stage1.step()

        if (i + 1) % args.log_interval == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs_stage1}], "
                  f"Iter [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")


def train_open_set(model, train_loader, optimizer_stage2, epoch, args):
    model.train()
    model.encoder.eval()
    model.classifier.eval()

    alpha = args.alpha

    for i, (images, labels) in enumerate(train_loader):
        if args.cuda:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer_stage2.zero_grad()
        x_recon_match, x_recon_nonmatch = model(images, labels, mode="stage2")

        loss_match = torch.mean(torch.sum(torch.abs(images - x_recon_match), dim=[1, 2, 3]))
        loss_nonmatch = torch.mean(torch.sum(torch.abs(images - x_recon_nonmatch), dim=[1, 2, 3]))
        loss = alpha * loss_match - (1 - alpha) * loss_nonmatch

        loss.backward()
        optimizer_stage2.step()

        if (i + 1) % args.log_interval == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs_stage2}] (alpha={alpha:.2f}), "
                  f"Iter [{i + 1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f} "
                  f"(Match: {loss_match.item():.4f}, NonMatch: {loss_nonmatch.item():.4f})")


def test_classification(model, test_known_loader, args):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_known_loader:
            if args.cuda:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(images, mode="stage1")
            all_outputs.append(outputs)
            all_labels.append(labels)

    preds_tensor = torch.cat(all_outputs)
    labels_tensor = torch.cat(all_labels)

    print(f"\n[Classification-Test] bs={args.batch_size}, lr={args.lr}, "
          f"alpha={args.alpha}, k={args.num_known_classes}, cuda={args.cuda}")
    results = evaluate(preds_tensor, pred_u=None, labels=labels_tensor, use_gpu=args.cuda)
    if "ACC" in results:
        print(f"ACC={results['ACC']:.2f}%")
    return results


def test_prethreshold(model, train_loader, args):
    model.eval()
    match_errors = []
    nonmatch_errors = []

    with torch.no_grad():
        for images, labels in train_loader:
            if args.cuda:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            z = model.encoder(images)

            condition_match = model.create_condition_vector(labels)
            z_match = model.film(z, condition_match)
            x_recon_match = model.decoder(z_match)
            error_match = torch.sum(torch.abs(images - x_recon_match), dim=[1, 2, 3])
            match_errors.append(error_match)

            labels_nm = torch.randint(0, model.num_classes, labels.size(), device=labels.device)
            mask = labels_nm == labels
            labels_nm[mask] = (labels_nm[mask] + 1) % model.num_classes
            condition_nonmatch = model.create_condition_vector(labels_nm)

            z_nm = model.film(z, condition_nonmatch)
            x_recon_nm = model.decoder(z_nm)
            error_nm = torch.sum(torch.abs(images - x_recon_nm), dim=[1, 2, 3])
            nonmatch_errors.append(error_nm)

    match_tensor = torch.cat(match_errors).cpu().numpy()
    nonmatch_tensor = torch.cat(nonmatch_errors).cpu().numpy()
    return match_tensor, nonmatch_tensor


def test_open_set(model, test_known_loader, test_unknown_loader, threshold, args):
    model.eval()
    all_labels = []
    all_recon_errors = []

    with torch.no_grad():
        for images, _ in test_known_loader:
            if args.cuda:
                images = images.cuda(non_blocking=True)
            _, min_recon_error = model(images, mode="stage3")
            all_labels.append(torch.ones(images.size(0), device=images.device, dtype=torch.int))
            all_recon_errors.append(min_recon_error)

        for images, _ in test_unknown_loader:
            if args.cuda:
                images = images.cuda(non_blocking=True)
            _, min_recon_error = model(images, mode="stage3")
            all_labels.append(torch.zeros(images.size(0), device=images.device, dtype=torch.int))
            all_recon_errors.append(min_recon_error)

    labels_tensor = torch.cat(all_labels)
    recon_errors_tensor = torch.cat(all_recon_errors)

    known_scores = -recon_errors_tensor[labels_tensor == 1]
    unknown_scores = -recon_errors_tensor[labels_tensor == 0]

    print(f"\n[OpenSet-Test] bs={args.batch_size}, lr={args.lr}, alpha={args.alpha}, "
          f"k={args.num_known_classes}, cuda={args.cuda}, tau={float(threshold):.6f}")
    results = evaluate(known_scores, pred_u=unknown_scores, labels=None, use_gpu=args.cuda)
    if "AUROC" in results:
        print(f"AUROC={results['AUROC']:.4f}")
    if "F1" in results:
        print(f"F1={results['F1']:.4f}")
    return results


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    set_seed(args.seed, args.cuda)

    # 数据与模型
    known_classes = list(range(args.num_known_classes))
    dataset = CIFAR10_c2ae(
        known_classes=known_classes,
        batch_size=args.batch_size,
        train_root=os.path.join(args.data_dir, "train"),
        test_root=os.path.join(args.data_dir, "test"),
    )
    train_loader = dataset.train_loader
    test_known_loader = dataset.known_test_loader
    test_unknown_loader = dataset.unknown_test_loader

    model = C2AE(latent_dim=128, num_classes=args.num_known_classes)
    if args.cuda:
        model.cuda()
    print(model)

    # 损失与优化器
    criterion_stage1 = nn.CrossEntropyLoss()
    optimizer_stage1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=args.lr,
    )
    optimizer_stage2 = torch.optim.Adam(
        list(model.film.parameters()) + list(model.decoder.parameters()),
        lr=args.lr,
    )

    # 闭集分类训练 + 分类准确率测试
    for epoch in range(args.epochs_stage1):
        train_closed_set(model, train_loader, optimizer_stage1, criterion_stage1, epoch, args)
        if (epoch + 1) % 5 == 0:
            test_classification(model, test_known_loader, args)

    # 开集识别训练
    for epoch in range(args.epochs_stage2):
        train_open_set(model, train_loader, optimizer_stage2, epoch, args)

    # 先行测试（阈值估计）
    S_m, S_nm = test_prethreshold(model, train_loader, args)
    threshold = calculate_threshold(S_m, S_nm, p_u=0.5)

    # 开集测试
    test_open_set(model, test_known_loader, test_unknown_loader, threshold, args)


if __name__ == "__main__":
    main()
