import torch
from torchmetrics import AUROC, F1Score, Metric


class metric(Metric):
    full_state_update = False

    def __init__(self, num_classes, metric_list=None, average="macro", device=None):
        super().__init__()
        self.metric_list = (
            {m.strip().upper() for m in metric_list if m and m.strip()}
            if metric_list
            else {"F1", "AUROC"}
        )
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes, average=average)
        self.auroc = AUROC(task="binary")
        self._has_open = False
        if device:
            self.to(device)

    def update(self, preds, targets, open_scores=None, open_targets=None):
        preds_t = torch.as_tensor(preds)
        targets_t = torch.as_tensor(targets)
        self.f1_score.update(preds_t, targets_t)
        if open_scores is not None and open_targets is not None:
            self.auroc.update(torch.as_tensor(open_scores), torch.as_tensor(open_targets))
            self._has_open = True

    def compute(self):
        results = {}
        if "F1" in self.metric_list:
            results["f1"] = float(self.f1_score.compute().item())
        if "AUROC" in self.metric_list and self._has_open:
            results["auroc"] = float(self.auroc.compute().item())
        return results

    def reset(self):
        super().reset()
        self.f1_score.reset()
        self.auroc.reset()
        self._has_open = False

    def print_results(self, results: dict, header: str | None = None):
        if not results:
            return
        parts = []
        for k, v in results.items():
            parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        msg = " | ".join(parts)
        print(f"{header} | {msg}" if header else msg)
