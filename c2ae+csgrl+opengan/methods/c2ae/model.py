import torch
import torch.nn as nn

from libs.metrics import metric
from methods.c2ae.network import C2AE


class c2ae:
    def __init__(self, args, latent_dim=128):
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
        self.model = C2AE(latent_dim=latent_dim, num_classes=args.num_known_classes).to(self.device)
        self.ce = nn.CrossEntropyLoss()
        self.opt_stage1 = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.classifier.parameters()), lr=args.lr)
        self.opt_stage2 = torch.optim.Adam(list(self.model.film.parameters()) + list(self.model.decoder.parameters()), lr=args.lr)
        self.alpha = args.alpha

    def train_stage1(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.opt_stage1.zero_grad()
            logits = self.model(images, mode="stage1")
            loss = self.ce(logits, labels)
            loss.backward()
            self.opt_stage1.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"c2ae stage1 epoch {epoch + 1}/{self.args.epochs_stage1} loss {avg_loss:.4f}")

    def train_stage2(self, loader, epoch):
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = False
        self.model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.opt_stage2.zero_grad()
            err_match, err_nonmatch = self.model(images, labels, mode="stage2")
            loss = self.alpha * err_match.mean() - (1 - self.alpha) * err_nonmatch.mean()
            loss.backward()
            self.opt_stage2.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"c2ae stage2 epoch {epoch + 1}/{self.args.epochs_stage2} loss {avg_loss:.4f}")

    def train(self, train_loader):
        for epoch in range(self.args.epochs_stage1):
            self.train_stage1(train_loader, epoch)
        for epoch in range(self.args.epochs_stage2):
            self.train_stage2(train_loader, epoch)

    def test(self, test_loader):
        known_loader, unknown_loader = test_loader
        self.model.eval()
        scores = []
        targets = []
        known_scores = []
        with torch.no_grad():
            for images, _ in known_loader:
                images = images.to(self.device)
                logits = self.model(images, mode="stage1")
                _, recon = self.model(images, mode="stage3")
                prob = torch.softmax(logits, dim=1).max(dim=1).values
                recon_score = torch.sigmoid(-recon)
                score = prob * recon_score
                scores.append(score)
                targets.append(torch.ones_like(score, dtype=torch.long))
                known_scores.append(score)
            for images, _ in unknown_loader:
                images = images.to(self.device)
                logits = self.model(images, mode="stage1")
                _, recon = self.model(images, mode="stage3")
                prob = torch.softmax(logits, dim=1).max(dim=1).values
                recon_score = torch.sigmoid(-recon)
                score = prob * recon_score
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
