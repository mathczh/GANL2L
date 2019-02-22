# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

__all__ = ["SNConv2d"]

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u, _v


class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.weight.data.normal_(0,0.02)
        if bias:
            self.bias.data.fill_(0)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())
        self.register_buffer('v', torch.Tensor(1, in_channels*kernel_size[0]*kernel_size[1]).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u, _v = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        self.v.copy_(_v)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def project(self):
        pass

    def showOrthInfo(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        W = self.weight.data.view(outputSize,-1)
        sigma = torch.sum(F.linear(self.u, torch.transpose(W, 0, 1)) * self.v)
        W = W / sigma
        _, s, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max().item())
        print('mean:',s.mean().item())
        print('min :',s.min().item())
        print('var :',s.var().item())
        return s
