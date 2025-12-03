import torch
import libs.data as data
import methods.csgrl.net as net
import tqdm
import math
import libs.metrics as metrics


class csgrl:
    def __init__(self, args):
        self.args = args
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # 动态类别数计算（根据 convert_class 中非 -1 的唯一标签）
        self.classnum = self.args.num_known_classes

        opendata = data.Data(self.args)
        dataloader, _ = opendata.get_dataloader()
        trainloader = dataloader[0]
        # 主模块
        self.backbone = net.Backbone_main(self.classnum, args).cuda()  # 为选择输出，特征提取器和分类器均置于backbone_main里
        self.generator = net.Generator_Class(self.classnum, nc= self.backbone.output_dim).cuda()
        # loss
        self.crt = net.CSGRLCriterion(args.arch_type)
        self.crtG = net.CriterionG() 
        # lr
        lr_backbone = self.args.learn_rate
        lr_Generator = self.args.learn_rateG


        self.opt_backbone = torch.optim.SGD(self.backbone.parameters(), lr= lr_backbone, weight_decay=5e-4)
        self.opt_Generator = torch.optim.SGD(self.generator.parameters(), lr= lr_Generator, weight_decay=5e-4)
        self.scheduler_D = net.SimpleLrScheduler(
                   lr_backbone,
                    milestones = self.args.milestones,
                    lr_decay = self.args.lr_decay,
                    warmup_epochs = self.args.warmup_epoch,
                    steps_per_epoch = len(trainloader)
                )
        self.scheduler_G = net.SimpleLrScheduler(
                    lr_Generator,
                    milestones = self.args.milestones,
                    lr_decay = self.args.lr_decay,
                    warmup_epochs = self.args.warmup_epoch,
                    steps_per_epoch=len(trainloader)
                )
        
        # 评估
        self.metrics = metrics.Metrics(self.args, self.classnum)
        
    def train(self, trainloader, testloader):
        close_trainloader = trainloader[0]

        for e in range(self.args.epoch_num):
            train_results = self.train_epoch(close_trainloader, e)
            if e % self.args.test_interval == 0:
                test_results = self.test(trainloader, testloader)
            results = self.metrics.compute(train_results, test_results)
            self.metrics.print_osr_results(results, e)


    def test(self, trainloader, testloader):
        self.backbone.eval()

        test_results = {
        'predictions': [],
        'gts': [],
        'know_scores': [],
        'unknow_scores': []
    }

        with torch.no_grad():
            for d in tqdm.tqdm(testloader, leave= False):
                image = d[0].cuda(non_blocking=True)
                gt = d[1].cuda(non_blocking=True)

                x_a, close_er = self.backbone(image, train_unknown = True)

                
                # pred = self.crt(close_er, pred=True)
                # 误差分离
                pred = self.crt(close_er[:, :-1, :, :], pred=True)  # logits→argmax 在crt内部
                know_er = close_er[:, :-1]            # [B, num_class, H, W]
                unknow_er = close_er[:, -1]           # [B, H, W]

                # 每个样本取其预测类别对应的误差
                batch_idx = torch.arange(pred.shape[0], device=pred.device)
                know_max_er = know_er[batch_idx, pred]   # [B, H, W]

                # 计算平均误差分数（mean over spatial dims）
                know_score = know_max_er.mean(dim=[1, 2])    # [B]
                unknow_score = unknow_er.mean(dim=[1, 2])    # [B]

                test_results['predictions'].append(pred.detach())
                test_results['gts'].append(gt.detach())
                test_results['know_scores'].append(know_score.detach())
                test_results['unknow_scores'].append(unknow_score.detach())
            
            return test_results

        
    def train_epoch(self, dataloader, epoch):
        self.backbone.train()
        self.generator.train()

        train_results = {
        'predictions': [],
        'gts': [],
        'total_lossD': 0,
        'total_lossG': 0,
        'num_samples': 0
    }
        
        for i, data in enumerate(tqdm.tqdm(dataloader, leave= False)):
            image, label = data[0].cuda(), data[1].cuda()
            batch_size = image.size(0)
            train_results['num_samples'] += batch_size
            train_results['gts'].append(label)
        
            # ======================== 训练重建模型 ========================
            if (self.args.epoch_num - epoch) > 10:  #前期只训练 B + R 
                lossD, close_er = self.train_epoch_stage1(image, label, epoch, i)   
            else:
                # ======================== 训练生成模型 ========================
                lossD, batch_lossG, close_er = self.train_epoch_stage2(image, label, epoch, i)
                
                train_results['total_lossG'] += batch_lossG.item()
            # 结果存储
            train_results['total_lossD'] += lossD.item() * batch_size
            pred = self.crt(close_er, pred=True)
            train_results['predictions'].append(pred.detach())

        return train_results
        
    def train_epoch_stage1(self,image, label, epoch, i):
        self.opt_backbone.zero_grad()
        
        x, close_er = self.backbone(image, train_unknown = False)
        lossD = self.crt(close_er, label)

        lossD.backward()
        self.opt_backbone.step()

            # 更新 Backbone 学习率
        lrD = self.scheduler_D.get_lr(epoch, i)
        for g in self.opt_backbone.param_groups:
            g['lr'] = lrD
        return lossD, close_er
    
    def train_epoch_stage2(self,image, label, epoch, i ):
        batch_lossG = torch.tensor(0.0).cuda() 

        # 生成数据准备
        max_dis = [0] * self.classnum   # 初始化每个类别的最大距离数组
        noise = []
        for c in range(self.classnum):
            noise.append(torch.randn(math.ceil(self.args.batch_size / self.classnum), 100, 1, 1).cuda()) 
        
        # ======================== 训练生成模型 ========================  
        self.opt_Generator.zero_grad()

        gen_data, gen_label = self.generator(noise)  # 为每个类别生成噪声，通过生成器生成样本和标签
        x, close_er = self.backbone(image, train_unknown = False)  
        max_dis = net.class_maximum_distance(
            -close_er.reshape([close_er.shape[0], close_er.shape[1], -1]).mean(dim=2), label, self.classnum, max_dis)  # 计算并更新每个类别的最大距离
        gen_x, gen_close_er = self.backbone(gen_data, isgen=True, train_unknown = False)
        
        # loss
        lossG1 = self.crt(gen_close_er, gen_label)  # 分类损失
        score = -torch.squeeze(gen_close_er)
        lossG2 = self.crtG(score, gen_label, max_dis, self.args.margin)  #  基于距离的特定损失
        lossG = lossG1 + lossG2

        # 更新学习率
        lrG = self.scheduler_G.get_lr(epoch, i)
        for g in self.opt_Generator.param_groups:
            g['lr'] = lrG

        lossG.backward()
        self.opt_Generator.step()

        # ======================== 训练重建模型 ========================
        self.opt_backbone.zero_grad()

        gen_data, gen_label = self.generator(noise)
        gen_x, gen_close_er = self.backbone(gen_data.detach(), isgen=True ,train_unknown = True )
        
        # loss
        lossD1 = self.crt(close_er, label)
        lossD2 = self.crt(gen_close_er, (torch.ones(gen_label.shape[0]) * self.classnum).cuda())
        lossD = lossD1 + lossD2
        lossD.backward()
        self.opt_backbone.step()

        # 更新 Backbone 学习率
        lrD = self.scheduler_D.get_lr(epoch, i)
        for g in self.opt_backbone.param_groups:
            g['lr'] = lrD
        batch_lossG += lossG.item() * self.args.batch_size  

        return lossD, batch_lossG, close_er

        

