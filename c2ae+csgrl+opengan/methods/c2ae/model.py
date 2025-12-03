import torch
import torch.nn as nn

from libs.metrics import Metrics
from libs.utils import calculate_threshold, compute_reconstruction_errors
from methods.c2ae.network import C2AE


class C2AEMethod():
    def __init__(self, args, latent_dim=128):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.metrics_to_display = getattr(args, "metrics_to_display", [])

        self.model = C2AE(latent_dim=latent_dim, num_classes=args.num_known_classes)
        if args.cuda:
            self.model.cuda()

        self.criterion_stage1 = nn.CrossEntropyLoss()
        self.optimizer_stage1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.classifier.parameters()),
            lr=args.lr
        )
        self.optimizer_stage2 = torch.optim.Adam(
            list(self.model.film.parameters()) + list(self.model.decoder.parameters()),
            lr=args.lr
        )

        self.stage1_accuracy = 0.0
        self.openset_results = None

    def print_metrics(self, metrics, header=None):
        if not metrics:
            return

        if self.metrics_to_display:
            metrics = {k: v for k, v in metrics.items() if k in self.metrics_to_display}

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

    def freeze_stage1_modules(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = False

    def set_decoder_bn_eval(self):
        for m in self.model.decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train_epoch_stage1(self, train_loader, epoch):
        args = self.args
        self.model.train()
        self.model.decoder.eval()

        total_loss = 0.0
        num_batches = len(train_loader)

        for images, labels in train_loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            self.optimizer_stage1.zero_grad()
            outputs = self.model(images, mode='stage1')
            loss = self.criterion_stage1(outputs, labels)
            loss.backward()
            self.optimizer_stage1.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        self.print_metrics(
            {"AvgLoss": avg_loss},
            header=f"Stage1 Epoch {epoch+1}/{args.epochs_stage1}"
        )
        return avg_loss

    def test_stage1(self, test_loader):
        args = self.args
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if args.cuda:
                    images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images, mode='stage1')
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / max(total, 1)
        self.stage1_accuracy = accuracy
        self.print_metrics({"Accuracy": accuracy}, header="Stage1 Test")
        return accuracy

    def train_stage1(self, train_loader, test_loader):
        args = self.args
        for epoch in range(args.epochs_stage1):
            self.train_epoch_stage1(train_loader, epoch)
            if (epoch + 1) % 5 == 0:
                self.test_stage1(test_loader)
        print("[Stage1] Training finished.")

    def train_epoch_stage2(self, train_loader, epoch):
        args = self.args
        self.model.train()
        self.model.encoder.eval()
        self.model.classifier.eval()
        self.set_decoder_bn_eval()
        alpha = args.alpha

        total_loss = 0.0
        total_match = 0.0
        total_nonmatch = 0.0
        num_batches = len(train_loader)

        for images, labels in train_loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            self.optimizer_stage2.zero_grad()
            err_match, err_nonmatch_min = self.model(images, labels, mode='stage2')

            loss_match = err_match.mean()
            loss_nonmatch = err_nonmatch_min.mean()
            loss = alpha * loss_match - (1 - alpha) * loss_nonmatch
            loss.backward()
            self.optimizer_stage2.step()

            total_loss += loss.item()
            total_match += loss_match.item()
            total_nonmatch += loss_nonmatch.item()

        avg_loss = total_loss / max(num_batches, 1)
        avg_match = total_match / max(num_batches, 1)
        avg_nonmatch = total_nonmatch / max(num_batches, 1)
        self.print_metrics(
            {
                "Loss": avg_loss,
                "MatchLoss": avg_match,
                "NonMatchLoss": avg_nonmatch
            },
            header=f"Stage2 Epoch {epoch+1}/{args.epochs_stage2}"
        )
        return avg_loss

    def train_stage2(self, train_loader):
        args = self.args
        self.freeze_stage1_modules()
        for epoch in range(args.epochs_stage2):
            self.train_epoch_stage2(train_loader, epoch)
        print("[Stage2] Training finished.")

    def test_openset(self, test_known_loader, test_unknown_loader):
        args = self.args
        self.model.eval()
        all_true_labels = []
        all_cls_preds = []
        all_recon_errors = []
        known_count = 0
        unknown_count = 0
        unknown_label = args.num_known_classes

        with torch.no_grad():
            for images, labels in test_known_loader:
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                preds, min_recon_error = self.model(images, mode='stage3')
                batch_size = images.size(0)
                known_count += batch_size
                all_true_labels.extend(labels.cpu().numpy().tolist())
                all_cls_preds.extend(preds.cpu().numpy().tolist())
                all_recon_errors.extend(min_recon_error.cpu().numpy().tolist())

            for images, _ in test_unknown_loader:
                if args.cuda:
                    images = images.cuda()
                preds, min_recon_error = self.model(images, mode='stage3')
                batch_size = images.size(0)
                unknown_count += batch_size
                all_true_labels.extend([unknown_label] * batch_size)
                all_cls_preds.extend(preds.cpu().numpy().tolist())
                all_recon_errors.extend(min_recon_error.cpu().numpy().tolist())

        self.print_metrics(
            {"KnownSamples": known_count, "UnknownSamples": unknown_count},
            header="Stage3 Data"
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
        统一训练 / 测试流程，便于通过 main.py 调用。
          - train_loader: 训练集 dataloader
          - test_loader:  (test_known_loader, test_unknown_loader)
        """
        if isinstance(test_loader, (list, tuple)) and len(test_loader) == 2:
            test_known_loader, test_unknown_loader = test_loader
        else:
            raise ValueError(
                "C2AEMethod.train expects test_loader=(test_known_loader, test_unknown_loader)"
            )

        print("\n" + "=" * 60)
        print("Stage 1: Closed-set Classification Training")
        print("=" * 60)
        self.train_stage1(train_loader, test_known_loader)

        print("\n" + "=" * 60)
        print("Stage 2: Open-set Training with Modified Non-Match Loss")
        print("=" * 60)
        self.train_stage2(train_loader)

        print("\n" + "=" * 60)
        print("Computing Reconstruction Errors with EVT Modeling")
        print("=" * 60)

        S_m, S_nm = compute_reconstruction_errors(self.model, train_loader, self.args)
        self.print_metrics(
            {"Size": len(S_m), "Mean": float(S_m.mean()), "Std": float(S_m.std())},
            header="Match errors S_m",
        )
        self.print_metrics(
            {"Size": len(S_nm), "Mean": float(S_nm.mean()), "Std": float(S_nm.std())},
            header="Non-match errors S_nm",
        )

        threshold = calculate_threshold(S_m, S_nm, p_u=0.5)
        self.print_metrics({"Threshold": threshold}, header="Optimal threshold")

        print("\n" + "=" * 60)
        print("Stage 3: Open-set Testing (k-inference)")
        print("=" * 60)

        results = self.test_openset(test_known_loader, test_unknown_loader)
        metrics_evaluator = Metrics(self.args)
        osr_metrics = metrics_evaluator.compute(
            results["true_labels"],
            results["cls_preds"],
            results["errors"],
            threshold,
        )
        Metrics.print_results(osr_metrics, header="Stage3 Evaluation")
