import torch
from torchmetrics import AUROC, F1Score, Metric


class metric(Metric):
    def __init__(self, metric_list=None, threshold=0.5, *args, **kwargs):
        super().__init__()
        self.metric_list = (
            {m.strip().upper() for m in metric_list} if metric_list else {"F1", "AUROC"}
        )
        self.threshold = threshold
        self.f1_score = F1Score(task="binary")
        self.auroc = AUROC(task="binary")

    def update(self, preds, targets):
        preds_t = torch.as_tensor(preds, dtype=torch.float32)
        targets_t = torch.as_tensor(targets, dtype=torch.long)
        bin_preds = (preds_t > self.threshold).long()
        self.f1_score.update(bin_preds, targets_t)
        self.auroc.update(preds_t, targets_t)

    def compute(self):
        results = {}
        if "F1" in self.metric_list:
            results["F1"] = self.f1_score.compute()
        if "AUROC" in self.metric_list:
            results["AUROC"] = self.auroc.compute()
        return results

    def reset(self):
        self.f1_score.reset()
        self.auroc.reset()

    def print_results(self):
        result = self.compute()
        for key, value in result.items():
            print(f"{key}:{value.item():.4f}")
