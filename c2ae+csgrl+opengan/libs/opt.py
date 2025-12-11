import argparse
import torch


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Open-set Recognition")
        self.parser.add_argument("--alpha", type=float, default=0.9, metavar="ALPHA")
        self.parser.add_argument("--arch_type", type=str, default="softmax_avg")
        self.parser.add_argument("--batch-size", type=int, default=128, metavar="N")
        self.parser.add_argument("--classification_hidden_dim", type=int, default=256)
        self.parser.add_argument("--data-dir", type=str, default="./data/cifar10", metavar="DIR")
        self.parser.add_argument("--epoch_num", type=int, default=50)
        self.parser.add_argument("--epochs-stage1", type=int, default=1, metavar="N")
        self.parser.add_argument("--epochs-stage2", type=int, default=10, metavar="N")
        self.parser.add_argument("--gamma", type=float, default=0.1)
        self.parser.add_argument("--learn_rate", type=float, default=0.4)
        self.parser.add_argument("--learn_rateG", type=float, default=0.4)
        self.parser.add_argument("--log-interval", type=int, default=50, metavar="N")
        self.parser.add_argument("--lr", type=float, default=0.0003, metavar="LR")
        self.parser.add_argument("--lr_decay", type=float, default=0.1)
        self.parser.add_argument("--margin", type=int, default=5)
        self.parser.add_argument("--metrics-to-display", type=str, default="all", metavar="NAME", help='comma separated metrics to print (use "all" for everything)')
        self.parser.add_argument("--metrics_to_track", type=list, default=["accuracy", "f1_score"])
        self.parser.add_argument("--milestones", type=list, default=[30, 45])
        self.parser.add_argument("--model_name", type=str, default="csgrl", help="model to run: baseline | c2ae | csgrl | opengan")
        self.parser.add_argument("--no-cuda", action="store_true", default=False)
        self.parser.add_argument("--num-known-classes", type=int, default=6, metavar="K", dest="num_known_classes")
        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--s_w", type=float, default=0.2)
        self.parser.add_argument("--seed", type=int, default=1, metavar="S")
        self.parser.add_argument("--task_type", type=str, default="OSR")
        self.parser.add_argument("--total-num-classes", type=int, default=10, dest="total_num_classes")
        self.parser.add_argument("--warmup_epoch", type=int, default=0)
        self.parser.add_argument("--test_interval", type=int, default=1)
        self.args = None

    def parse(self):
        self.args = self.parser.parse_args()
        self.args.cuda = not getattr(self.args, "no_cuda", False) and torch.cuda.is_available()
        raw = getattr(self.args, "metrics_to_display", "all")
        if isinstance(raw, str) and raw.strip().lower() == "all":
            self.args.metrics_to_display = []
        elif isinstance(raw, str):
            self.args.metrics_to_display = [m.strip() for m in raw.split(",") if m.strip()]
        else:
            self.args.metrics_to_display = list(raw) if raw else []
        return self.args

    def get_args(self):
        if self.args is None:
            return self.parse()
        return self.args
