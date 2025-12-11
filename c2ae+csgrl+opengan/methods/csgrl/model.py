import math
import torch
import tqdm
import methods.csgrl.network as net
from libs.loss import class_maximum_distance, CriterionG
from torchmetrics.functional.classification import binary_roc
from methods.basemodel import basemodel

class csgrl(basemodel):
    def __init__(self, args):
        super().__init__(args)
        # self.args = args
        # self.device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
        # self.classnum = args.num_known_classes

        self.backbone = net.Backbone_main(self.classnum, args).to(self.device)
        self.generator = net.Generator_Class(self.classnum, nc=self.backbone.output_dim).to(self.device)

        # self.crt = torch.nn.CrossEntropyLoss()
        self.crtG = CriterionG()

        self.opt_backbone = torch.optim.SGD(self.backbone.parameters(), lr=args.learn_rate, weight_decay=5e-4)
        self.opt_Generator = torch.optim.SGD(self.generator.parameters(), lr=args.learn_rateG, weight_decay=5e-4)

        # self.metric = metric(self.args, metric_list=getattr(self.args, "metrics_to_track", []))

    def train_epoch_stage1(self, loader, epoch):
        self.backbone.train()
        total_loss = 0.0
        for i, data in enumerate(tqdm.tqdm(loader, leave= False)):
            image, label = data[0].to(self.device), data[1].to(self.device)

            self.opt_backbone.zero_grad()
            _, logit, _ = self.backbone(image, train_unknown=False)
            lossD = self.crt(logit, label)
            lossD.backward()
            self.opt_backbone.step()

            total_loss += lossD.item()
        avg = total_loss / max(len(loader), 1)
        print(f"csgrl | train | stage1 | epoch: [{epoch + 1}/{self.args.epochs_stage1}] | lossD: {avg:.4f}")

    def train_epoch_stage2(self, loader, epoch):
        self.backbone.train()
        self.generator.train()
        total_lossD = 0.0
        total_lossG = 0.0
        for i, data in enumerate(tqdm.tqdm(loader, leave= False)):
            image, label = data[0].to(self.device), data[1].to(self.device)
            batch_size = image.size(0)
            max_dis = [0] * self.classnum
            noise = [torch.randn(math.ceil(self.args.batch_size / self.classnum), 100, 1, 1, device=self.device) for _ in range(self.classnum)]
            
            # 生成器训练
            self.opt_Generator.zero_grad()
            gen_feat, gen_label = self.generator(noise)
            close_error, logit, pred = self.backbone(image, train_unknown=False)
            max_dis = class_maximum_distance(-close_error.reshape([close_error.shape[0], close_error.shape[1], -1]).mean(dim=2), label, self.classnum, max_dis)
            gen_close_error, gen_logit, _ = self.backbone(gen_feat, isgen=True, train_unknown=False)
            lossG1 = self.crt(gen_logit, gen_label.long())
            score = -torch.squeeze(gen_close_error)
            lossG2 = self.crtG(score, gen_label, max_dis, self.args.margin)
            lossG = lossG1 + lossG2
            lossG.backward()
            self.opt_Generator.step()
            
            # 分类器训练
            self.opt_backbone.zero_grad()
            gen_data, gen_label = self.generator(noise)
            gen_close_error, gen_logit, _ = self.backbone(gen_data.detach(), isgen=True, train_unknown=True)
            lossD1 = self.crt(logit, label)
            lossD2 = self.crt(gen_logit, gen_label.long())
            lossD = lossD1 + lossD2
            lossD.backward()
            self.opt_backbone.step()

            total_lossD += lossD.item() * batch_size
            total_lossG += lossG.item() * batch_size
        avgD = total_lossD / max(len(loader) * batch_size, 1)
        avgG = total_lossG / max(len(loader) * batch_size, 1)
        print(f"csgrl | train | stage2 | epoch: [{epoch + 1}/{self.args.epochs_stage2}] | lossD: {avgD:.4f} | lossG: {avgG:.4f}")

    # def train(self, train_loader, test_loader):

    #     for epoch in range(self.args.epochs_stage1):
    #         self.train_epoch_stage1(train_loader, epoch)
    #         if epoch % self.args.test_interval == self.args.test_interval - 1:
    #             self.test(test_loader)
    #     for epoch in range(self.args.epochs_stage2):
    #         self.train_epoch_stage2(train_loader, epoch)
    #         if epoch % self.args.test_interval == self.args.test_interval - 1:
    #             self.test(test_loader)

    def test(self, test_loader):
        self.backbone.eval()
        self.generator.eval()
        
        gts = []
        predictions = []
        know_scores = []
        unknow_scores = []

        with torch.no_grad():
            for d in tqdm.tqdm(test_loader, leave= False):
                x1 = d[0].cuda(non_blocking=True)
                gt = d[1].cuda(non_blocking=True)

                close_error, logit, pred = self.backbone(x1, train_unknown = True, is_test = True)

        # ========== 计算本 epoch 的指标 ===========
                # 误差分离
                know_er = close_error[:, :-1]            # [B, num_class, H, W]
                unknow_er = close_error[:, -1]           # [B, H, W]

                # 每个样本取其预测类别对应的误差
                batch_idx = torch.arange(pred.shape[0], device=pred.device)
                know_max_er = know_er[batch_idx, pred]   # [B, H, W]

                # 计算平均误差分数（mean over spatial dims）
                know_score = know_max_er.mean(dim=[1, 2])    # [B]
                unknow_score = unknow_er.mean(dim=[1, 2])    # [B]

                know_scores.append(know_score)
                unknow_scores.append(unknow_score)
                predictions.append(pred)
                gts.append(gt)

        # 拼接为整体张量
        gts = torch.cat(gts)
        mask = gts < 6
        predictions = torch.cat(predictions)
        know_scores = torch.cat(know_scores)
        unknow_scores = torch.cat(unknow_scores)

        self.metric.update(predictions[mask], gts[mask], "acc")
        # 标准化
        know_mean, know_std = know_scores.mean(), know_scores.std()
        unknow_mean, unknow_std = unknow_scores.mean(), unknow_scores.std()
        know_scores_stand = (know_scores - know_mean) / (know_std + 1e-8)
        unknow_scores_stand = (unknow_scores - unknow_mean) / (unknow_std + 1e-8)

        # 计算多种 s_w 权重下的 close_probs
        labels = (gts < 6).int()
        w_tensor = torch.tensor(self.args.s_w, device=know_scores.device, dtype=know_scores.dtype)
        close_prob_j = (1 - w_tensor) * know_scores_stand - (w_tensor * unknow_scores_stand)

        fpr, tpr, thresholds = binary_roc(close_prob_j, labels)
        tidx = torch.argmin(torch.abs(tpr - 0.95))
        thresh = thresholds[tidx]
        preds = (close_prob_j >= thresh).int()
    
        self.metric.update(preds, labels, "F1")
        self.metric.print_results()

    # def main(self, train_loader, test_loader):
    #     self.train(train_loader, test_loader)
    #     self.test(test_loader)
