import torch
from torchmetrics import AUROC, F1Score, Metric, Accuracy


class metric(Metric):
    def __init__(self, args, metric_list=None,  **kwargs):
        super().__init__()
        self.args = args
        self.metric_list = (
            {m.strip().upper() for m in metric_list} if metric_list else {"F1", "AUROC"}
        )
        self.f1_score = F1Score(task="binary").cuda()
        self.auroc = AUROC(task="binary").cuda()
        self.acc = Accuracy(task="multiclass" , num_classes=self.args.num_known_classes).cuda()

    def update(self, preds, targets, metric_type):
        if metric_type == "F1":
            self.f1_score.update(preds, targets)
        elif metric_type == "acc":
            self.acc.update(preds, targets)
        elif metric_type == "auroc":
            self.auroc.update(preds, targets)  

    def compute(self):
        results = {}
        if "F1_SCORE" in self.metric_list:
            results["F1"] = self.f1_score.compute()
        if "ACCURACY" in self.metric_list:
            results["ACC"] = self.acc.compute()
        if "AUROC" in self.metric_list:
            results["AUROC"] = self.auroc.compute()
        return results

    def reset(self):
        self.f1_score.reset()
        self.acc.reset()
        self.auroc.reset()

    def print_results(self):
        result = self.compute()
        for i, (key, value) in enumerate(result.items()):
            if i == 0:
                print("        test | ", end= "")
            if i < len(result) - 1:
                print(f"{key}: {value.item():.4f}", end= " | ")
            else:
                print(f"{key}: {value.item():.4f}")
        self.reset()
