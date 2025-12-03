import numpy as np
import torch
import torch.nn as nn

from libs.metrics import Metrics
from libs.utils import calculate_threshold
from methods.baseline.network import BaselineNet


class BaselineMethod:
    """
    Reconstruction-based open-set baseline for comparison with C2AE.

    - Stage 1: closed-set classification training (encoder + classifier).
    - Stage 2: autoencoder training (encoder + decoder) on known-class data.
    - Stage 3: open-set testing based on reconstruction error.
    """

    def __init__(self, args, latent_dim: int = 128):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.metrics_to_display = getattr(args, "metrics_to_display", [])

        self.model = BaselineNet(
            latent_dim=latent_dim, num_classes=args.num_known_classes
        ).to(self.device)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_recon = nn.L1Loss()

        self.optimizer_stage1 = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.classifier.parameters()),
            lr=args.lr,
        )
        self.optimizer_stage2 = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters()),
            lr=args.lr,
        )

        self.stage1_accuracy = 0.0
        self.openset_results = None

    def print_metrics(self, metrics, header=None):
        if not metrics:
            return

        if self.metrics_to_display:
            metrics = {
                k: v for k, v in metrics.items() if k in self.metrics_to_display
            }

        if not metrics:
            return

        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")

        message = ", ".join(parts)
        if header:
            print(f"{header} | {message}")
        else:
            print(message)

    def train_epoch_stage1(self, train_loader, epoch: int):
        args = self.args
        self.model.train()

        total_loss = 0.0
        num_batches = len(train_loader)

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer_stage1.zero_grad()

            logits = self.model(images, mode="classify")
            loss = self.criterion_cls(logits, labels)
            loss.backward()
            self.optimizer_stage1.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        self.print_metrics(
            {"AvgLoss": avg_loss},
            header=f"[Baseline] Stage1 Epoch {epoch + 1}/{args.epochs_stage1}",
        )
        return avg_loss

    def test_stage1(self, test_loader):
        args = self.args
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images, mode="classify")
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / max(total, 1)
        self.stage1_accuracy = accuracy
        self.print_metrics({"Accuracy": accuracy}, header="[Baseline] Stage1 Test")
        return accuracy

    def train_stage1(self, train_loader, test_loader):
        args = self.args
        for epoch in range(args.epochs_stage1):
            self.train_epoch_stage1(train_loader, epoch)
            if (epoch + 1) % 5 == 0:
                self.test_stage1(test_loader)
        print("[Baseline][Stage1] Training finished.")

    def train_epoch_stage2(self, train_loader, epoch: int):
        args = self.args
        self.model.train()

        total_loss = 0.0
        num_batches = len(train_loader)

        for images, _ in train_loader:
            images = images.to(self.device)

            self.optimizer_stage2.zero_grad()
            x_recon = self.model(images, mode="reconstruct")
            loss = self.criterion_recon(x_recon, images)
            loss.backward()
            self.optimizer_stage2.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        self.print_metrics(
            {"ReconLoss": avg_loss},
            header=f"[Baseline] Stage2 Epoch {epoch + 1}/{args.epochs_stage2}",
        )
        return avg_loss

    def train_stage2(self, train_loader):
        args = self.args
        for epoch in range(args.epochs_stage2):
            self.train_epoch_stage2(train_loader, epoch)
        print("[Baseline][Stage2] Training finished.")

    def test_openset(self, test_known_loader, test_unknown_loader):
        """
        Open-set evaluation with reconstruction error + classifier:
        - true_labels: 多类别标签（0..K-1 为已知类，K 为未知类）
        - cls_preds: 分类器在 closed-set 上的预测（0..K-1）
        - errors: 每个样本的重构误差
        """
        self.model.eval()
        all_true_labels = []
        all_cls_preds = []
        all_recon_errors = []
        known_count = 0
        unknown_count = 0
        unknown_label = self.args.num_known_classes

        with torch.no_grad():
            for images, labels in test_known_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images, mode="classify")
                _, preds = torch.max(logits.data, 1)

                recon = self.model(images, mode="reconstruct")
                error = torch.mean(torch.abs(images - recon), dim=[1, 2, 3]).cpu().numpy()

                batch_size = images.size(0)
                known_count += batch_size
                all_true_labels.extend(labels.cpu().numpy().tolist())
                all_cls_preds.extend(preds.cpu().numpy().tolist())
                all_recon_errors.extend(error.tolist())

            for images, _ in test_unknown_loader:
                images = images.to(self.device)

                logits = self.model(images, mode="classify")
                _, preds = torch.max(logits.data, 1)

                recon = self.model(images, mode="reconstruct")
                error = torch.mean(torch.abs(images - recon), dim=[1, 2, 3]).cpu().numpy()

                batch_size = images.size(0)
                unknown_count += batch_size
                all_true_labels.extend([unknown_label] * batch_size)
                all_cls_preds.extend(preds.cpu().numpy().tolist())
                all_recon_errors.extend(error.tolist())

        self.print_metrics(
            {"KnownSamples": known_count, "UnknownSamples": unknown_count},
            header="[Baseline] Stage3 Data",
        )

        self.openset_results = {
            "true_labels": all_true_labels,
            "cls_preds": all_cls_preds,
            "errors": all_recon_errors,
        }
        return self.openset_results

    # 统一入口：在 main.py 中通过 phase='train' 调用
    def train(self, train_loader, test_loader):
        """
        统一训练 / 测试流程，便于通过 main.py 调用：
          - train_loader: 训练集 dataloader
          - test_loader:  (test_known_loader, test_unknown_loader)
        """
        if isinstance(test_loader, (list, tuple)) and len(test_loader) == 2:
            test_known_loader, test_unknown_loader = test_loader
        else:
            raise ValueError(
                "BaselineMethod.train expects test_loader=(test_known_loader, test_unknown_loader)"
            )

        # Stage 1 & 2
        print("\n" + "=" * 60)
        print("[Baseline] Stage 1: Closed-set Classification Training")
        print("=" * 60)
        self.train_stage1(train_loader, test_known_loader)

        print("\n" + "=" * 60)
        print("[Baseline] Stage 2: Autoencoder Training")
        print("=" * 60)
        self.train_stage2(train_loader)

        # 在训练集上估计阈值（简单做法：基于 reconstruction error 的分位数）
        print("\n" + "=" * 60)
        print("[Baseline] Estimating threshold from training reconstruction errors")
        print("=" * 60)

        self.model.eval()
        train_errors = []
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(self.device)
                recon = self.model(images, mode="reconstruct")
                error = torch.mean(
                    torch.abs(images - recon), dim=[1, 2, 3]
                ).cpu().numpy()
                train_errors.extend(error.tolist())

        train_errors = np.array(train_errors)
        # 近似：将训练误差视为匹配分布，构造一个简单的“非匹配”分布（加噪）喂给阈值函数
        S_m = train_errors
        S_nm = train_errors + np.random.normal(
            loc=train_errors.mean(),
            scale=max(train_errors.std(), 1e-6),
            size=train_errors.shape,
        )
        threshold = float(calculate_threshold(S_m, S_nm, p_u=0.5))
        self.print_metrics({"Threshold": threshold}, header="[Baseline] Threshold")

        # Stage 3: Open-set testing
        print("\n" + "=" * 60)
        print("[Baseline] Stage 3: Open-set Testing")
        print("=" * 60)

        results = self.test_openset(test_known_loader, test_unknown_loader)
        metrics_evaluator = Metrics(self.args)
        osr_metrics = metrics_evaluator.compute(
            results["true_labels"],
            results["cls_preds"],
            results["errors"],
            threshold,
        )
        Metrics.print_results(osr_metrics, header="[Baseline] Stage3 Evaluation")
