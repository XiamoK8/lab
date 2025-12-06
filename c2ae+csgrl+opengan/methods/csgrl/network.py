import math
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
import time
from torchvision.transforms import transforms
import logging
import os
from torchmetrics.classification import AUROC, F1Score

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual


    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        # block: 残差块类型 ,nb_layers: 该模块中包含的残差块数量 ,activate_before_residual: 是否在残差连接前激活
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
   
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    

class WideResNetBackbone(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropRate=0.0,req_output_dim = -1):
        super(WideResNetBackbone, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.output_dim = nChannels[3]
        if req_output_dim > 0 and req_output_dim != self.output_dim:
            self.dim_map = nn.Sequential(
                nn.Conv2d(self.output_dim, req_output_dim, 1,1, 0, bias=False),
                nn.BatchNorm2d(req_output_dim),
                nn.LeakyReLU(0.2))
            self.output_dim = req_output_dim
        else:
            self.dim_map = None
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        if self.dim_map is not None:
            out = self.dim_map(out)
        return out


def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res


class AutoEncoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan

        if latent_chan > 0: # 自编码器
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h, )
                dcv = layer_block(h, inchannel, use_activation=i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)

            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()  # 解码器层应该与编码器相反
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel, latent_chan)  # 编码器最后一层,将特征压缩到潜在空间
            self.latent_deconv = layer_block(latent_chan, inchannel, use_activation=(len(hidden_layers) > 0))
        else:
            self.center = nn.Parameter(torch.rand([inchannel, 1, 1]), True)

    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output, latent   #返回重建结果和潜在编码
        else:
            return self.center, self.center


class CSGRLClassifier(nn.Module):

    def __init__(self, inchannels, num_class, args, ae_hidden = [], ae_latent = 64 ):
        super().__init__()   

        # 创建 num_class + 1 个自编码器
        self.args = args
        self.class_aes = []
        for i in range(num_class + 1):  
            ae = AutoEncoder(inchannels, ae_hidden, ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)

        self.reduction = -1 * self.args.gamma

    def forward(self, feat, train_unknown = False):
        cls_errors = []
        autoencoder_num = len(self.class_aes) if train_unknown else len(self.class_aes) - 1
        for i in range(autoencoder_num):
            rec_feat, mid_feat = self.class_aes[i](feat)
            cls_error = torch.norm(rec_feat - feat, p=1, dim=1, keepdim=True) * self.reduction
            #　Ｌ1距离 |重建 - 原始| * (-0.1)  确保经过softmax后 误差越小的概率越高， 0.1：控制概率转换的"硬度"，值越小概率分布越平滑
            cls_error = torch.clamp(cls_error, -100, 100)  # 将误差 限制在 [-100, 100], 为防止极端值
            cls_errors.append(cls_error)
        logits = torch.cat(cls_errors, dim=1)
        return logits


class Backbone_main(nn.Module):

    def __init__(self, num_classes, args):
        super().__init__()
        self.args = args
        self.backbone = WideResNetBackbone( 40, 4, 0, -1)  # 特征提取器
        self.output_dim = self.backbone.output_dim
        self.cat_cls = CSGRLClassifier(self.backbone.output_dim, num_classes, self.args)  # 分类器

    def forward(self, x, feature_only=False, isgen=False, train_unknown = False):
        if not isgen:
            x = self.backbone(x)
        if feature_only:  # 非生成数据才通过 backbone 提取特征
            return x
        xcls_raw = self.cat_cls(x, train_unknown )  # 分类输出
        # x_com = self.reco(x)
        return x, xcls_raw


class CSGRLCriterion(nn.Module):

    def get_onehot_label(self, y, clsnum):
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)  
    #　在维度1（列维度）上，根据y中的索引位置填充1，实现one-hot编码

    def __init__(self, avg_order):
        super().__init__()
        self.avg_order = {"avg_softmax": 1, "softmax_avg": 2}[avg_order]
        # 1: 先平均池化再softmax   , 2: 先softmax再平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x, y=None, prob=False, pred=False):
        if self.avg_order == 1:
            g = self.avg_pool(x).view(x.shape[0], -1)
            g = torch.softmax(g, dim=1)
        elif self.avg_order == 2:
            g = torch.softmax(x, dim=1)
            g = self.avg_pool(g).view(x.size(0), -1)
        if prob: return g  # 概率
        if pred: return torch.argmax(g, dim=1)  # 最大值作为预测结果
        loss = -torch.sum(self.get_onehot_label(y, g.shape[1]) * torch.log(g), dim=1).mean()
        # if torch.isinf(loss) or torch.isnan(loss):
        #     print(1)
        return loss  # 交叉熵损失


class CriterionG(nn.Module):

    def __init__(self):
        super().__init__()
        self.sig =  nn.Sigmoid()

    def forward(self, close_er, y, max_dis, margin):
        loss = 0
        j = 0
        for i in range(len(max_dis)):  # 遍历每个类别
            index = torch.where(y==i)[0]  #找到当前批次中属于类别i的样本索引
            if len(index) == 0:
                continue
            gap = self.sig(close_er[index,i] - max_dis[i] - margin)  
            # close_er[index,i］所有该类别样本对该类的重建误差
            gap = torch.clamp(gap,1e-7,1-1e-7)  #　差距值限制在[1e-7, 1-1e-7]范围内，防止出现０或１
            loss += -torch.log(gap).mean()
            j += 1   # 实际处理的类别数量
        loss /= j
        return loss

    
class Generator_Class(nn.Module):
    def __init__(self, num_class, nz=100, ngf=64, nc=512):  
        # nz=100: 噪声向量的维度 , nc=512: 输出通道数（生成的特征维度）,ngf=64: 生成器特征图数量
        super(Generator_Class, self).__init__()
        self.class_gen = []
        for i in range(num_class):
            gen = Generator(nz, ngf, nc).cuda()
            self.class_gen.append(gen)
        self.class_gen = nn.ModuleList(self.class_gen)   #将列表转换为　ModuleList

    def forward(self, x):
        cls_gens = []  
        cls_genls = []
        for i in range(len(self.class_gen)):
            if len(x[i]) ==0:  # 检查输入是否为空
                continue
            cls_gen = self.class_gen[i](x[i])
            cls_gens.append(cls_gen)
            cls_genl = (torch.ones(cls_gen.shape[0])*i).cuda()  # 创建与生成数据数量相同的标签张量,全为i
            cls_genls.append(cls_genl)

        gendata = torch.cat(cls_gens, dim=0)
        genlabel = torch.cat(cls_genls, dim=0)
        return gendata, genlabel
    
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=512):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(self.nz, self.ngf * 8, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d(self.ngf * 4, self.nc, 1, 1, 0, bias=True),
            # nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        if input.shape[0] == 1:  # 避免批次大小为1时的除零问题
            for m in self.main.modules():
                if isinstance(m, nn.BatchNorm2d):  # 将 batchnorm设置为评估模式, 会使用训练时的均值和方差
                    m.eval()
            output = self.main(input)
            for m in self.main.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
            return output
        return self.main(input)


class SimpleLrScheduler:
    """
    支持 warmup + multi-step decay
    """
    def __init__(self, init_lr, milestones=None, lr_decay=0.1, warmup_epochs=0, steps_per_epoch=1):
        self.init_lr = init_lr
        self.milestones = milestones or []
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.steps_per_epoch = steps_per_epoch

    def get_lr(self, epoch, step):
        tstep = epoch * self.steps_per_epoch + step
        # ---- Warmup ----
        if self.warmup_steps > 0 and tstep <= self.warmup_steps:
            return self.init_lr * tstep / max(1, self.warmup_steps)
        # ---- Multi-step decay ----
        lr = self.init_lr
        for m in self.milestones:
            if epoch >= m:
                lr *= self.lr_decay
        return lr

def class_maximum_distance( cls_er, y, clsnum, max_dis):
    for i in range(clsnum):
        index = torch.where(y == i)[0]
        if index.numel() == 0:
            continue
        temp = torch.max(cls_er[index, i]).detach()  #计算当前批次中类别i的最大值：
        if max_dis[i] < temp:
            max_dis[i] = temp

    return max_dis
    
def setup_logger(log_file):
    """
    初始化日志系统：控制台 + 文件同时输出
    会立即创建日志文件（即使还没写入内容）
    """
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        # 创建日志文件夹
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        #  确保日志文件存在（立即创建）
        open(log_file, 'a', encoding='utf-8').close()

        # 文件输出
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加两个 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 初始日志信息
        logger.info(f"Logger initialized. Writing logs to: {os.path.abspath(log_file)}")

    return logger
alph_pre = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
           1]




