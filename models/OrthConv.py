'''
Usage
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules import Module
from torch.nn import functional as F
from torch.autograd import Variable
import math

__all__ = ['Orth_Plane_Conv2d','Orth_Plane_Mani_Conv2d','Orth_UV_Conv2d','Orth_UV_Mani_Conv2d','GroupOrthConv']

class ManiGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_out):
        input, = ctx.saved_tensors
        input = Variable(input)

        originSize = input.size()
        outputSize = originSize[0]
        W = input.view(outputSize, -1)
        Wt = torch.t(W)
        WWt = W.mm(Wt)

        d_p = grad_out.view(outputSize, -1)

        # Version1: WWtG-WGtW
        d_p = (WWt.mm(d_p) - W.mm(d_p.t()).mm(W))
        # Version2: G - WGtW
        # d_p = d_p - W.mm(d_p.t()).mm(W)

        grad_in = d_p.view(originSize)
        return grad_in

mani_grad = ManiGrad.apply

class Orth_Plane_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                norm=False, w_norm=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Orth_Plane_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.total_in_dim = in_channels*kernel_size[0]*kernel_size[1]
        if out_channels  > self.total_in_dim:
            raise ValueError('out_channels must not be greater than input dimension (in_channels*kernel_size[0]*kernel_size[1])')

        self.eps = 1e-8
        self.norm = norm
        self.w_norm = w_norm
        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if bias:
            self.bias.data.fill_(0)
        self.projectiter = 0
        self.project(style='qr', interval = 1)

    def forward(self, input):
        _weight = self.weight
        _input = input
        # if self.w_norm:
        #     _weight = _weight/ torch.norm(_weight.view(self.out_channels,-1),2,1).clamp(min = self.eps).view(-1,1,1,1)

        _output = F.conv2d(input, _weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(_input**2, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

        return _output

    def orth_penalty(self):
        originSize = self.weight.size()
        outputSize = originSize[0]
        W = self.weight.view(outputSize, -1)
        Wt = torch.t(W)
        WWt = W.mm(Wt)
        I = Variable(torch.eye(WWt.size()[0]).cuda(), requires_grad=False)
        return ((WWt.sub(I))**2).sum()

    def project(self, style='qr', interval = 1):
        '''
        Project weight to l2 ball
        '''
        self.projectiter = self.projectiter+1
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        if style=='qr' and self.projectiter%interval == 0:
            # Compute the qr factorization
            q, r = torch.qr(self.weight.data.view(outputSize,-1).t())
            # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            self.weight.data = q.t().view(originSize)
        elif style=='svd' and self.projectiter%interval == 0:
            """
            Problematic
            """
            # Compute the svd factorization (may be not stable)
            u, s, v = torch.svd(self.weight.data.view(outputSize,-1))
            self.weight.data = u.mm(v.t()).view(originSize)
        elif self.w_norm:
            self.weight.data =  self.weight.data/ torch.norm(self.weight.data.view(outputSize,-1),2,1).clamp(min = 1e-8).view(-1,1,1,1)

    def showOrthInfo(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        W = self.weight.data.view(outputSize,-1)
        _, s, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max().item())
        print('mean:',s.mean().item())
        print('min :',s.min().item())
        print('var :',s.var().item())
        print('penalty :', ((W.mm(W.t())-torch.eye(outputSize).cuda())**2).sum().item() )
        return s

class Orth_Plane_Mani_Conv2d(Orth_Plane_Conv2d):
    def forward(self, input):

        _weight = mani_grad(self.weight)
        _input = input
        _output = F.conv2d(input, _weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(_input**2, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

        return _output

class Orth_UV_Conv2d(Module):
    '''
    W = UdV
    Mode 1: ! divided by max
    Mode 2: ! truncate by 1
    Mode 3: ! penalize sum of all log max spectral
    Mode 4: divided by max and then clip to [0.5,1]
    Mode 5: penalize E(-log(q(x))) q(x)~|N(0,0.2)| & sum of all log max spectral (fail to directly apply)
    Mode 6: ! penalize E(-log(q(x))) q(x)~|N(0,0.2)| & divided by max
    Mode 7: ! penalize E(-log(q(x))) q(x)~|N(0,0.2)| & truncate by 1 (worked)
    Mode 8: ! penalize dlogd & divided by max
    Mode 9: penalize expd & divided by max
    Mode 10: penalize logd & divided by max
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, norm = False):
        self.eps = 1e-8
        self.norm = norm

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Orth_UV_Conv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.total_in_dim = in_channels*kernel_size[0]*kernel_size[1]
        self.weiSize = (self.out_channels,in_channels,kernel_size[0],kernel_size[1])

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = _pair(0)
        self.groups = groups

        if self.out_channels  <= self.total_in_dim:
            self.Uweight = Parameter(torch.Tensor(self.out_channels, self.out_channels))
            self.Dweight = Parameter(torch.Tensor(self.out_channels))
            self.Vweight = Parameter(torch.Tensor(self.out_channels, self.total_in_dim))
            self.Uweight.data.normal_(0, math.sqrt(2. / self.out_channels))
            self.Vweight.data.normal_(0, math.sqrt(2. / self.total_in_dim))
            self.Dweight.data.fill_(1)
        else:
            self.Uweight = Parameter(torch.Tensor(self.out_channels, self.total_in_dim))
            self.Dweight = Parameter(torch.Tensor(self.total_in_dim))
            self.Vweight = Parameter(torch.Tensor(self.total_in_dim, self.total_in_dim))
            self.Uweight.data.normal_(0, math.sqrt(2. / self.out_channels))
            self.Vweight.data.normal_(0, math.sqrt(2. / self.total_in_dim))
            self.Dweight.data.fill_(1)
        self.projectiter = 0
        self.project(style='qr', interval = 1)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
            self.bias.data.fill_(0)
        else:
            self.register_parameter('bias', None)

        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

    def setmode(self, mode):
        self.mode = mode

    def update_sigma(self):
        if self.mode in (1,6,8,9,10):
            self.Dweight.data = self.Dweight.data/self.Dweight.data.abs().max()
        elif self.mode in (2,7):
            self.Dweight.data.clamp_(-1, 1)
        elif self.mode == 4:
            self.Dweight.data = self.Dweight.data/self.Dweight.data.abs().max()
            self.Dweight.data.clamp_(0.4, 1)

    def log_spectral(self):
        return torch.log(self.Dweight.abs().max())

    def spectral_penalty(self):
        if self.mode in (5,6,7):
            if(len(self.Dweight)==1):
                return 0
            sd2 = 0.1**2
            _d, _ = self.Dweight.sort()
            return ( (1 - _d[:-1])**2/sd2-torch.log((_d[1:] - _d[:-1])+1e-8) ).mean()
        elif self.mode == 8:
            return (self.Dweight*torch.log(self.Dweight)).mean()
        elif self.mode == 9:
            return (torch.exp(self.Dweight)).mean()
        elif self.mode == 10:
            return -(torch.log(self.Dweight)).mean()
        else:
            raise RuntimeError("error mode")

    @property
    def W_(self):
        self.update_sigma()
        return self.Uweight.mm(self.Dweight.diag()).mm(self.Vweight).view(self.weiSize)

    def forward(self, input):
        _output = F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return _output

    def orth_penalty(self):
        penalty = 0

        if self.out_channels  <= self.total_in_dim:
            W = self.Uweight
        else:
            W = self.Uweight.t()
        Wt = torch.t(W)
        WWt = W.mm(Wt)
        I = Variable(torch.eye(WWt.size()[0]).cuda())
        penalty = penalty+((WWt.sub(I))**2).sum()


        W = self.Vweight
        Wt = torch.t(W)
        WWt = W.mm(Wt)
        I = Variable(torch.eye(WWt.size()[0]).cuda())
        penalty = penalty+((WWt.sub(I))**2).sum()
        return penalty

    def project(self, style='none', interval = 1):
        '''
        Project weight to l2 ball
        '''
        self.projectiter = self.projectiter+1
        if style=='qr' and self.projectiter%interval == 0:
            # Compute the qr factorization for U
            if self.out_channels  <= self.total_in_dim:
                q, r = torch.qr(self.Uweight.data.t())
            else:
                q, r = torch.qr(self.Uweight.data)
            # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            if self.out_channels  <= self.total_in_dim:
                self.Uweight.data = q.t()
            else:
                self.Uweight.data = q

            # Compute the qr factorization for V
            q, r = torch.qr(self.Vweight.data.t())
            # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            self.Vweight.data = q.t()
        elif style=='svd' and self.projectiter%interval == 0:
            # Compute the svd factorization (may be not stable) for U
            u, s, v = torch.svd(self.Uweight.data)
            self.Uweight.data = u.mm(v.t())

            # Compute the svd factorization (may be not stable) for V
            u, s, v = torch.svd(self.Vweight.data)
            self.Vweight.data = u.mm(v.t())

    def showOrthInfo(self):
        s= self.Dweight.data
        _D = self.Dweight.data.diag()
        W = self.Uweight.data.mm(_D).mm(self.Vweight.data)
        _, ss, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max().item(),'max* :',ss.max().item())
        print('mean:',s.mean().item(),'mean*:',ss.mean().item())
        print('min :',s.min().item(),'min* :',ss.min().item())
        print('var :',s.var().item(),'var* :',ss.var().item())
        print('s RMSE: ', ((s-ss)**2).mean().item()**0.5)
        if self.out_channels  <= self.total_in_dim:
            pu = (self.Uweight.data.mm(self.Uweight.data.t())-torch.eye(self.Uweight.size()[0]).cuda()).norm().item()**2
        else:
            pu = (self.Uweight.data.t().mm(self.Uweight.data)-torch.eye(self.Uweight.size()[1]).cuda()).norm().item()**2
        pv =  (self.Vweight.data.mm(self.Vweight.data.t())-torch.eye(self.Vweight.size()[0]).cuda()).norm().item()**2
        print('penalty :', pu, ' (U) + ', pv, ' (V)' )
        return ss


class Orth_UV_Mani_Conv2d(Orth_UV_Conv2d):
    def forward(self, input):
        #_weight = mani_grad(self.Uweight).mm(self.Dweight.diag()).mm(mani_grad(self.Vweight)).view(self.weiSize)
        _weight = self.Uweight.mm(self.Dweight.diag()).mm(self.Vweight).view(self.weiSize)
        _input = input
        # if self.w_norm:
        #     _weight = _weight/ torch.norm(_weight.view(self.out_channels,-1),2,1).clamp(min = self.eps).view(-1,1,1,1)

        _output = F.conv2d(input, _weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.norm:
            input_norm = torch.sqrt(F.conv2d(_input**2, Variable(self.input_norm_wei), None,
                                self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
            _output = _output/input_norm

        return _output

class GroupOrthConv(nn.Module):
    '''
    devide output channels into 'groups'
    '''
    def __init__(self, Orth_Conv2d, in_channels, out_channels, kernel_size,
                stride=1, padding=0, bias=False, groups=None):
        super(GroupOrthConv, self).__init__()
        if groups == None:
            groups = (out_channels-1)//(in_channels*kernel_size*kernel_size)+1
        self.groups = groups
        self.gourp_out_channels = np.ones(groups) * (out_channels//groups)
        if out_channels%groups > 0:
            self.gourp_out_channels[:out_channels%groups] += 1
        self.sconvs = []
        for i in range(groups):
            newsconv = Orth_Conv2d(in_channels, int(self.gourp_out_channels[i]),
                            kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)
            self.add_module('sconv{0}'.format(i),newsconv)
            self.sconvs.append(newsconv)

    def forward(self,x):
        out = []
        for i in range(self.groups):
            out.append(self.sconvs[i](x))
        return torch.cat(out,1)
