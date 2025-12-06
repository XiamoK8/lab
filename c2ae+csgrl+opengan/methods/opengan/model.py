import torch
import torch.nn as nn
import torch.optim as optim

from libs.metrics import metric
from methods.opengan.network import FeatureExtractor, Classifier, Generator, Discriminator


class opengan:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
        self.num_known_classes = args.num_known_classes
        self.F = FeatureExtractor().to(self.device)
        self.C = Classifier(num_classes=self.num_known_classes).to(self.device)
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCELoss()
        self.opt_C = optim.SGD(list(self.F.parameters()) + list(self.C.parameters()), lr=self.args.lr * 3)
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

    def train_stage1(self, loader, epoch):
        self.F.train()
        self.C.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            feats = self.F(imgs)
            logits = self.C(feats)
            loss = self.loss_cls(logits, labels)
            self.opt_C.zero_grad()
            loss.backward()
            self.opt_C.step()
            total_loss += loss.item()
        avg = total_loss / max(len(loader), 1)
        print(f"opengan stage1 epoch {epoch + 1}/{self.args.epochs_stage1} loss {avg:.4f}")

    def train_stage2(self, loader, unknown_loader, epoch):
        self.F.eval()
        for p in self.F.parameters():
            p.requires_grad = False
        self.D.train()
        self.G.train()
        lossD_list = []
        lossG_list = []
        open_iter = iter(unknown_loader)
        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            bsz = imgs.size(0)
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
                open_iter = iter(unknown_loader)
                open_imgs, _ = next(open_iter)
            open_imgs = open_imgs.to(self.device)
            with torch.no_grad():
                feats_open = self.F(open_imgs).view(open_imgs.size(0), 512, 1, 1)
            out_open = self.D(feats_open)
            loss_open = self.loss_bce(out_open, torch.zeros_like(out_open))
            loss_D = loss_real + loss_open + loss_fake
            loss_D.backward()
            self.opt_D.step()
            self.opt_G.zero_grad()
            z = torch.randn(bsz, 100, 1, 1, device=self.device)
            feats_fake = self.G(z)
            out_fake_g = self.D(feats_fake)
            loss_G = self.loss_bce(out_fake_g, torch.ones_like(out_fake_g))
            loss_G.backward()
            self.opt_G.step()
            lossD_list.append(loss_D.item())
            lossG_list.append(loss_G.item())
        avgD = sum(lossD_list) / max(len(lossD_list), 1)
        avgG = sum(lossG_list) / max(len(lossG_list), 1)
        print(f"opengan stage2 epoch {epoch + 1}/{self.args.epochs_stage2} lossD {avgD:.4f} lossG {avgG:.4f}")

    def train(self, train_loader, test_loader):
        known_loader, unknown_loader = test_loader
        for epoch in range(self.args.epochs_stage1):
            self.train_stage1(train_loader, epoch)
        for epoch in range(self.args.epochs_stage2):
            self.train_stage2(train_loader, unknown_loader, epoch)

    def test(self, test_loader):
        known_loader, unknown_loader = test_loader
        self.D.eval()
        self.C.eval()
        self.F.eval()
        scores = []
        targets = []
        known_scores = []
        with torch.no_grad():
            for imgs, _ in known_loader:
                imgs = imgs.to(self.device)
                feats = self.F(imgs)
                logits_cls = self.C(feats)
                probs_cls = torch.softmax(logits_cls, dim=1).max(dim=1).values
                open_score = torch.sigmoid(self.D(feats.view(feats.size(0), 512, 1, 1))).view(-1)
                score = probs_cls * open_score
                scores.append(score)
                targets.append(torch.ones_like(score, dtype=torch.long))
                known_scores.append(score)
            for imgs, _ in unknown_loader:
                imgs = imgs.to(self.device)
                feats = self.F(imgs)
                logits_cls = self.C(feats)
                probs_cls = torch.softmax(logits_cls, dim=1).max(dim=1).values
                open_score = torch.sigmoid(self.D(feats.view(feats.size(0), 512, 1, 1))).view(-1)
                score = probs_cls * open_score
                scores.append(score)
                targets.append(torch.zeros_like(score, dtype=torch.long))
        scores = torch.cat(scores)
        targets = torch.cat(targets)
        thr = torch.median(torch.cat(known_scores)) if known_scores else torch.tensor(0.5, device=self.device)
        m = metric(metric_list=getattr(self.args, "metrics_to_display", []), threshold=float(thr)).to(self.device)
        m.update(scores, targets)
        m.print_results()

    def main(self, train_loader, test_loader):
        self.train(train_loader, test_loader)
        self.test(test_loader)
