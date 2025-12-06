import math
import torch

import methods.csgrl.network as net
from libs.metrics import metric


class csgrl:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
        self.classnum = args.num_known_classes
        self.backbone = net.Backbone_main(self.classnum, args).to(self.device)
        self.generator = net.Generator_Class(self.classnum, nc=self.backbone.output_dim).to(self.device)
        self.crt = net.CSGRLCriterion(args.arch_type)
        self.crtG = net.CriterionG()
        self.opt_backbone = torch.optim.SGD(self.backbone.parameters(), lr=args.learn_rate, weight_decay=5e-4)
        self.opt_Generator = torch.optim.SGD(self.generator.parameters(), lr=args.learn_rateG, weight_decay=5e-4)
        self.scheduler_D = None
        self.scheduler_G = None

    def train_epoch_stage1(self, loader, epoch):
        self.backbone.train()
        total_loss = 0.0
        for i, data in enumerate(loader):
            image, label = data[0].to(self.device), data[1].to(self.device)
            self.opt_backbone.zero_grad()
            _, close_er = self.backbone(image, train_unknown=False)
            lossD = self.crt(close_er, label)
            lossD.backward()
            self.opt_backbone.step()
            lrD = self.scheduler_D.get_lr(epoch, i)
            for g in self.opt_backbone.param_groups:
                g["lr"] = lrD
            total_loss += lossD.item()
        avg = total_loss / max(len(loader), 1)
        print(f"csgrl stage1 epoch {epoch + 1}/{self.args.epochs_stage1} loss {avg:.4f}")

    def train_epoch_stage2(self, loader, epoch):
        self.backbone.train()
        self.generator.train()
        total_lossD = 0.0
        total_lossG = 0.0
        for i, data in enumerate(loader):
            image, label = data[0].to(self.device), data[1].to(self.device)
            batch_size = image.size(0)
            max_dis = [0] * self.classnum
            noise = [torch.randn(math.ceil(self.args.batch_size / self.classnum), 100, 1, 1, device=self.device) for _ in range(self.classnum)]
            self.opt_Generator.zero_grad()
            gen_data, gen_label = self.generator(noise)
            _, close_er = self.backbone(image, train_unknown=False)
            max_dis = net.class_maximum_distance(-close_er.reshape([close_er.shape[0], close_er.shape[1], -1]).mean(dim=2), label, self.classnum, max_dis)
            _, gen_close_er = self.backbone(gen_data, isgen=True, train_unknown=False)
            lossG1 = self.crt(gen_close_er, gen_label)
            score = -torch.squeeze(gen_close_er)
            lossG2 = self.crtG(score, gen_label, max_dis, self.args.margin)
            lossG = lossG1 + lossG2
            lrG = self.scheduler_G.get_lr(epoch, i)
            for g in self.opt_Generator.param_groups:
                g["lr"] = lrG
            lossG.backward()
            self.opt_Generator.step()
            self.opt_backbone.zero_grad()
            gen_data, gen_label = self.generator(noise)
            _, gen_close_er = self.backbone(gen_data.detach(), isgen=True, train_unknown=True)
            lossD1 = self.crt(close_er, label)
            lossD2 = self.crt(gen_close_er, (torch.ones(gen_label.shape[0], device=self.device) * self.classnum))
            lossD = lossD1 + lossD2
            lrD = self.scheduler_D.get_lr(epoch, i)
            for g in self.opt_backbone.param_groups:
                g["lr"] = lrD
            lossD.backward()
            self.opt_backbone.step()
            total_lossD += lossD.item() * batch_size
            total_lossG += lossG.item() * self.args.batch_size
        avgD = total_lossD / max(len(loader) * self.args.batch_size, 1)
        avgG = total_lossG / max(len(loader) * self.args.batch_size, 1)
        print(f"csgrl stage2 epoch {epoch + 1}/{self.args.epochs_stage2} lossD {avgD:.4f} lossG {avgG:.4f}")

    def train(self, train_loader, test_loader=None):
        self.scheduler_D = net.SimpleLrScheduler(self.args.learn_rate, milestones=self.args.milestones, lr_decay=self.args.lr_decay, warmup_epochs=self.args.warmup_epoch, steps_per_epoch=len(train_loader))
        self.scheduler_G = net.SimpleLrScheduler(self.args.learn_rateG, milestones=self.args.milestones, lr_decay=self.args.lr_decay, warmup_epochs=self.args.warmup_epoch, steps_per_epoch=len(train_loader))
        for epoch in range(self.args.epochs_stage1):
            self.train_epoch_stage1(train_loader, epoch)
        for epoch in range(self.args.epochs_stage2):
            self.train_epoch_stage2(train_loader, epoch)

    def test(self, test_loader):
        self.backbone.eval()
        scores = []
        targets = []
        known_loader, unknown_loader = test_loader
        with torch.no_grad():
            for images, _ in known_loader:
                images = images.to(self.device)
                _, close_er = self.backbone(images, train_unknown=True)
                pred = self.crt(close_er[:, :-1, :, :], pred=True)
                know_er = close_er[:, :-1]
                unknow_er = close_er[:, -1]
                batch_idx = torch.arange(pred.shape[0], device=pred.device)
                know_max_er = know_er[batch_idx, pred]
                know_score = know_max_er.mean(dim=[1, 2])
                unknow_score = unknow_er.mean(dim=[1, 2])
                scores.append(torch.cat([-know_score, -unknow_score], dim=0))
                targets.append(torch.cat([torch.ones_like(know_score, dtype=torch.long), torch.zeros_like(unknow_score, dtype=torch.long)], dim=0))
            for images, _ in unknown_loader:
                images = images.to(self.device)
                _, close_er = self.backbone(images, train_unknown=True)
                pred = self.crt(close_er[:, :-1, :, :], pred=True)
                know_er = close_er[:, :-1]
                unknow_er = close_er[:, -1]
                batch_idx = torch.arange(pred.shape[0], device=pred.device)
                know_max_er = know_er[batch_idx, pred]
                know_score = know_max_er.mean(dim=[1, 2])
                unknow_score = unknow_er.mean(dim=[1, 2])
                scores.append(torch.cat([-know_score, -unknow_score], dim=0))
                targets.append(torch.cat([torch.ones_like(know_score, dtype=torch.long), torch.zeros_like(unknow_score, dtype=torch.long)], dim=0))
        scores = torch.cat(scores)
        targets = torch.cat(targets)
        thr = torch.median(scores[targets == 1]) if (targets == 1).any() else torch.tensor(0.5, device=self.device)
        m = metric(metric_list=getattr(self.args, "metrics_to_display", []), threshold=float(thr)).to(self.device)
        m.update(scores, targets)
        m.print_results()

    def main(self, train_loader, test_loader):
        self.train(train_loader, test_loader)
        self.test(test_loader)
