import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from libs.metrics import Metrics
from methods.opengan.net import (
    FeatureExtractor,
    Classifier,
    Generator,
    Discriminator,
)


class OpenGAN:
    """
    Two-stage OpenGAN training:
      - Stage A: closed-set classifier pre-training.
      - Stage B: adversarial training with generated and open-set features.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.num_known_classes = getattr(opt, "num_known_classes", 0)
        self.unknown_label = self.num_known_classes

        # Network modules
        self.F = FeatureExtractor().to(self.device)
        self.C = Classifier(num_classes=self.num_known_classes).to(self.device)
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        # Losses
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCELoss()

        # Optimizers
        self.opt_C = optim.SGD(
            list(self.F.parameters()) + list(self.C.parameters()),
            lr=self.opt.lr * 3,
        )
        self.opt_G = optim.Adam(
            self.G.parameters(), lr=self.opt.lr, betas=(0.5, 0.999)
        )
        self.opt_D = optim.Adam(
            self.D.parameters(), lr=self.opt.lr, betas=(0.5, 0.999)
        )

        # Metrics helper
        self.metrics = Metrics(
            args=self.opt,
            num_classes=self.num_known_classes + 1,
            device=self.device,
        )

        # Data loaders will be provided at runtime
        self.closed_loader = None
        self.open_loader = None
        self.test_loader = None

        self.current_stage = None  # "A" or "B"

    # ======================================================================
    # Data loader helpers
    # ======================================================================
    def _set_loaders(self, train_loader=None, test_loader=None):
        if train_loader is not None:
            if not isinstance(train_loader, (list, tuple)) or len(train_loader) != 2:
                raise ValueError(
                    "OpenGAN expects train_loader=(closed_loader, open_loader)"
                )
            self.closed_loader, self.open_loader = train_loader
        if test_loader is not None:
            self.test_loader = test_loader

    # ======================================================================
    # Training entry
    # ======================================================================
    def train(self, train_loader=None, test_loader=None):
        self._set_loaders(train_loader, test_loader)

        if self.closed_loader is None or self.open_loader is None:
            raise ValueError(
                "OpenGAN.train requires train_loader=(closed_loader, open_loader)"
            )

        # Stage A: closed-set pretraining
        print("\n========== Stage A: Closed-set Pretraining ==========")
        self.current_stage = "A"
        for ep in range(self.opt.epochs_stage1):
            loss_avg = self.train_epoch()
            print(
                f"[Stage A] Epoch {ep + 1}/{self.opt.epochs_stage1}   Loss={loss_avg:.4f}"
            )

        # Freeze feature extractor
        for p in self.F.parameters():
            p.requires_grad = False
        self.F.eval()

        # Stage B: adversarial training
        print("\n========== Stage B: Adversarial Training ==========")
        self.current_stage = "B"
        for ep in range(self.opt.epochs_stage2):
            lossD, lossG = self.train_epoch()
            print(
                f"[Stage B] Epoch {ep + 1}/{self.opt.epochs_stage2}   "
                f"D_loss={lossD:.4f}  G_loss={lossG:.4f}"
            )

        if self.test_loader is not None:
            return self.test()
        return None

    # ======================================================================
    # Stage dispatcher
    # ======================================================================
    def train_epoch(self):
        if self.current_stage == "A":
            return self.train_epoch_stageA()
        elif self.current_stage == "B":
            return self.train_epoch_stageB()
        else:
            raise ValueError("current_stage must be either 'A' or 'B'")

    # ======================================================================
    # Stage A: closed-set classifier pretraining
    # ======================================================================
    def train_epoch_stageA(self):
        self.F.train()
        self.C.train()

        total_loss = 0

        for imgs, labels in self.closed_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            feats = self.F(imgs)
            logits = self.C(feats)
            loss = self.loss_cls(logits, labels)

            self.opt_C.zero_grad()
            loss.backward()
            self.opt_C.step()

            total_loss += loss.item()

        return total_loss / max(len(self.closed_loader), 1)

    # ======================================================================
    # Stage B: GAN adversarial training
    # ======================================================================
    def train_epoch_stageB(self):
        self.D.train()
        self.G.train()

        lossD_list, lossG_list = [], []
        open_iter = iter(self.open_loader)

        for imgs, _ in self.closed_loader:
            imgs = imgs.to(self.device)
            bsz = imgs.size(0)

            # Discriminator update: real + fake + open-set
            with torch.no_grad():
                feats_real = self.F(imgs).view(bsz, 512, 1, 1)

            self.opt_D.zero_grad()

            out_real = self.D(feats_real)
            loss_real = self.loss_bce(out_real, torch.ones_like(out_real))

            z = torch.randn(bsz, 100, 1, 1, device=self.device)
            feats_fake = self.G(z)
            out_fake = self.D(feats_fake.detach())
            loss_fake = self.loss_bce(out_fake, torch.zeros_like(out_fake))

            try:
                open_imgs, _ = next(open_iter)
            except StopIteration:
                open_iter = iter(self.open_loader)
                open_imgs, _ = next(open_iter)

            open_imgs = open_imgs.to(self.device)
            with torch.no_grad():
                feats_open = self.F(open_imgs).view(open_imgs.size(0), 512, 1, 1)
            out_open = self.D(feats_open)
            loss_open = self.loss_bce(out_open, torch.zeros_like(out_open))

            loss_D = loss_real + loss_open + loss_fake
            loss_D.backward()
            self.opt_D.step()

            # Generator update: fool the discriminator
            self.opt_G.zero_grad()

            z = torch.randn(bsz, 100, 1, 1, device=self.device)
            feats_fake = self.G(z)
            out_fake_g = self.D(feats_fake)
            loss_G = self.loss_bce(out_fake_g, torch.ones_like(out_fake_g))

            loss_G.backward()
            self.opt_G.step()

            lossD_list.append(loss_D.item())
            lossG_list.append(loss_G.item())

        return np.mean(lossD_list), np.mean(lossG_list)

    # ======================================================================
    # Testing: compute K+1 classification metrics
    # ======================================================================
    def test(self, train_loader=None, test_loader=None):
        self._set_loaders(train_loader, test_loader)

        if self.test_loader is None:
            raise ValueError("OpenGAN.test requires a test_loader.")

        self.D.eval()
        self.C.eval()
        self.F.eval()

        test_results = {"predictions": [], "gts": []}

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                feats = self.F(imgs)
                logits_cls = self.C(feats)
                probs_cls = torch.softmax(logits_cls, dim=1)

                open_score = torch.sigmoid(
                    self.D(feats.view(feats.size(0), 512, 1, 1))
                ).view(-1, 1)

                probs_kplus1 = torch.cat(
                    [probs_cls * open_score, 1 - open_score], dim=1
                )
                preds = torch.argmax(probs_kplus1, dim=1)

                test_results["predictions"].append(preds.cpu())
                test_results["gts"].append(labels.cpu())

        results = self.metrics.compute(test_results)

        print("\n===== Test Results =====")
        self.metrics.print_results(results, header="[OpenGAN]")
        return results
