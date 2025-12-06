import torch
import torch.nn as nn

from libs.metrics import metric
from methods.basemodel.network import BaseNet


class basemodel:
    def __init__(self, args, latent_dim=128):
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
        self.model = BaseNet(latent_dim=latent_dim, num_classes=args.num_known_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self, train_loader, test_loader):
        for epoch in range(self.args.epochs_stage1):
            self.model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"basemodel epoch {epoch + 1}/{self.args.epochs_stage1} loss {avg_loss:.4f}")

    def test(self, test_loader):
        known_loader, unknown_loader = test_loader
        self.model.eval()
        scores = []
        targets = []
        with torch.no_grad():
            for images, _ in known_loader:
                images = images.to(self.device)
                prob = torch.softmax(self.model(images), dim=1).max(dim=1).values
                scores.append(prob.cpu())
                targets.append(torch.ones_like(prob, dtype=torch.long).cpu())
            for images, _ in unknown_loader:
                images = images.to(self.device)
                prob = torch.softmax(self.model(images), dim=1).max(dim=1).values
                scores.append(prob.cpu())
                targets.append(torch.zeros_like(prob, dtype=torch.long).cpu())
        scores = torch.cat(scores)
        targets = torch.cat(targets)
        m = metric(metric_list=getattr(self.args, "metrics_to_display", []))
        m.update(scores, targets)
        m.print_results()

    def main(self, train_loader, test_loader):
        self.train(train_loader, test_loader)
        self.test(test_loader)
