#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MultiPat2
@File ：losses.py
@IDE  ：PyCharm 
@Author ：魏丙财
@Date ：2021/8/12 21:30 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        # loss = torch.sum(error) # 是mean还是sum
        loss = torch.mean(error)
        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        Laplacian = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Laplacian_3C = torch.Tensor(1, 3, 3, 3)
        Laplacian_3C[:, 0:3, :, :] = Laplacian
        self.conv_la = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_la.weight = torch.nn.Parameter(Laplacian_3C)

    def forward(self, X, Y):
        X_la = self.conv_la(X)
        Y_la = self.conv_la(Y)
        # compute gradient of Y
        self.conv_la.train(False)
        loss = F.mse_loss(X_la, Y_la, size_average=True)

        return loss
