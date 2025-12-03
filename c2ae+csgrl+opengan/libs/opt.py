import argparse
import torch


class CFG:
    def __init__(self):
        parser = self._build_parser()
        args = parser.parse_args()
        self.__dict__.update(vars(args))

        # runtime helpers
        self.cuda = not self.no_cuda and torch.cuda.is_available()

        # seeding
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        # metrics filter
        metrics_raw = getattr(self, "metrics_to_display", "all")
        if metrics_raw.strip().lower() == "all":
            self.metrics_to_display = []
        else:
            self.metrics_to_display = [
                metric.strip() for metric in metrics_raw.split(",") if metric.strip()
            ]

    def _build_parser(self):
        parser = argparse.ArgumentParser(description="Open-set Recognition")

        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            metavar="N",
            help="input batch size for training (default: 128)",
        )
        parser.add_argument(
            "--epochs-stage1",
            type=int,
            default=20,
            metavar="N",
            help="number of epochs for stage 1 (default: 20)",
        )
        parser.add_argument(
            "--epochs-stage2",
            type=int,
            default=50,
            metavar="N",
            help="number of epochs for stage 2 (default: 50)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.0003,
            metavar="LR",
            help="learning rate (default: 0.0003)",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.9,
            metavar="ALPHA",
            help="weight for match loss (default: 0.9)",
        )
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1,
            metavar="S",
            help="random seed (default: 1)",
        )
        parser.add_argument(
            "--total-num-classes",
            type=int,
            default=10,
            dest="total_num_classes",
        )
        parser.add_argument(
            "--num-known-classes",
            type=int,
            default=6,
            metavar="K",
            dest="num_known_classes",
            help="number of known classes (default: 6)",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=50,
            metavar="N",
            help="how many batches to wait before logging",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default="./data",
            metavar="DIR",
            help="directory for dataset",
        )
        parser.add_argument(
            "--metrics-to-display",
            type=str,
            default="all",
            metavar="NAME",
            help='comma separated metrics to print (use "all" for everything)',
        )

        # csgrl 训练参数
        parser.add_argument("--s_w", type=float, default=0.2)
        parser.add_argument("--learn_rate", type=float, default=0.4)
        parser.add_argument("--learn_rateG", type=float, default=0.4)
        parser.add_argument("--margin", type=int, default=5)
        parser.add_argument("--epoch_num", type=int, default=50)
        parser.add_argument("--lr_decay", type=float, default=0.1)
        parser.add_argument("--milestones", type=list, default=[30, 45])
        parser.add_argument("--warmup_epoch", type=int, default=0)

        # 模型架构参数
        parser.add_argument("--classification_hidden_dim", type=int, default=256)
        parser.add_argument("--arch_type", type=str, default="softmax_avg")

        # 分类模型 / 任务控制参数
        parser.add_argument(
            "--model_name",
            type=str,
            default="csgrl",
            help="model to run: baseline | c2ae | csgrl | opengan",
        )
        parser.add_argument(
            "--phase",
            type=str,
            default="train",
            help="method name of the chosen model to call",
        )
        parser.add_argument(
            "--task_type",
            type=str,
            default="OSR",
            help="reserved; not used for imports in this repo",
        )
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument(
            "--metrics_to_track",
            type=list,
            default=["accuracy", "f1_score"],
        )
        parser.add_argument("--test_interval", type=int, default=1)

        return parser


__all__ = ["CFG"]
