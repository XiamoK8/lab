import torch.nn as nn
import torch

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
    

def class_maximum_distance( cls_er, y, clsnum, max_dis):
    for i in range(clsnum):
        index = torch.where(y == i)[0]
        if index.numel() == 0:
            continue
        temp = torch.max(cls_er[index, i]).detach()  #计算当前批次中类别i的最大值：
        if max_dis[i] < temp:
            max_dis[i] = temp

    return max_dis