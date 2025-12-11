import torch
from libs.metrics import metric

class basemodel:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
        self.classnum = args.num_known_classes

        self.crt = torch.nn.CrossEntropyLoss()

        self.metric = metric(self.args, metric_list=getattr(self.args, "metrics_to_track", []))

    def train_epoch_stage1(self, loader, epoch):
        pass

    def train_epoch_stage2(self, loader, epoch):
        pass
    
    def train(self, train_loader, test_loader):

        for epoch in range(self.args.epochs_stage1):
            self.train_epoch_stage1(train_loader, epoch)
            if epoch % self.args.test_interval == self.args.test_interval - 1:
                self.test(test_loader)
        for epoch in range(self.args.epochs_stage2):
            self.train_epoch_stage2(train_loader, epoch)
            if epoch % self.args.test_interval == self.args.test_interval - 1:
                self.test(test_loader)

    def test(self, test_loader):
        pass

    def main(self, train_loader, test_loader):
        self.train(train_loader, test_loader)
        self.test(test_loader)