import torch.nn as nn
import torch
import torch.nn.functional as F
from ..OrthConv import *
from ..SNConv import *
from ..SphereConv import *

__all__ = ['resnet_G','resnet_D','set_use_bias_res']


USE_BIAS = False

def set_use_bias_res(x):
    global USE_BIAS
    USE_BIAS = x

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, myConv2d = nn.Conv2d, use_BN = False):
        super(ResBlockDiscriminator, self).__init__()

        modules = []
        if use_BN:
            modules.append(nn.BatchNorm2d(in_channels))
        modules.append(nn.ReLU())
        modules.append(myConv2d(in_channels, out_channels, 3, 1, padding=1, bias=USE_BIAS))
        if use_BN:
            modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
        modules.append(myConv2d(out_channels, out_channels, 3, 1, padding=1, bias=USE_BIAS))
        if stride > 1:
            modules.append(nn.AvgPool2d(2, stride=stride, padding=0))
        self.model = nn.Sequential(*modules)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                myConv2d(in_channels,out_channels, 1, 1, padding=0, bias=USE_BIAS),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, myConv2d = nn.Conv2d):
        super(FirstResBlockDiscriminator, self).__init__()

        if myConv2d is Orth_Plane_Conv2d:
            first_myConv2d = nn.Conv2d
        else:
            first_myConv2d = myConv2d

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            first_myConv2d(in_channels, out_channels, 3, 1, padding=1, bias=USE_BIAS),
            nn.ReLU(),
            myConv2d(out_channels, out_channels, 3, 1, padding=1, bias=USE_BIAS),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            first_myConv2d(in_channels, out_channels, 1, 1, padding=0, bias=USE_BIAS),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Generator(nn.Module):
    def __init__(self, isize, z_dim, nc, ngf, ngpu):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.ngpu = ngpu
        self.dense = nn.ConvTranspose2d(self.z_dim, 512, 4, stride=1, padding=0)
        self.final = nn.Conv2d(64, nc, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            self.dense,
            ResBlockGenerator(512, 256, stride=2),
            ResBlockGenerator(256, 128, stride=2),
            ResBlockGenerator(128, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, isize, nc, ndf, ngpu, norm_type = 'none', loss_type='wgan'):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # input is nc x isize x isize
        if 'OR' in norm_type:
            self.myConv2d = Orth_Plane_Conv2d
        elif 'UVR' in norm_type:
            self.myConv2d = Orth_UV_Conv2d
        elif 'WN' in norm_type:
            self.myConv2d = WN_Conv2d
        elif 'SN' in norm_type:
            self.myConv2d = SNConv2d
        elif 'Sphere' in norm_type:
            self.myConv2d = Sphere_Conv2d
        else:
            self.myConv2d = nn.Conv2d

        use_BN = 'BN' in norm_type

        self.main = nn.Sequential(
                FirstResBlockDiscriminator(nc, 64, stride=2, myConv2d = self.myConv2d),
                ResBlockDiscriminator(64, 128, stride=2, myConv2d = self.myConv2d, use_BN = use_BN),
                ResBlockDiscriminator(128, 256, stride=2, myConv2d = self.myConv2d, use_BN = use_BN),
                ResBlockDiscriminator(256, 512, stride=2, myConv2d = self.myConv2d, use_BN = use_BN),
                ResBlockDiscriminator(512, 512, myConv2d = self.myConv2d, use_BN = use_BN),
                nn.ReLU(),
                nn.AvgPool2d(4),
                self.myConv2d(512, 1, 1, 1, 0, bias=USE_BIAS)
            )
        if loss_type == "dcgan":
            self.main.add_module('final_sigmoid', nn.Sigmoid())

        for m_name, m in self.named_modules():
            print(m_name+': '+m.__class__.__name__)
            if isinstance(m, nn.Conv2d): # Special Convlayer has their own ini
                nn.init.xavier_uniform(m.weight.data, 1.)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    def showBNInfo(self):
        for m_name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                print("BN weight: max: %f; mean: %f; min: %f; var: %f" %( m.weight.data.max(), m.weight.data.mean(), m.weight.data.min(),m.weight.data.var()))
                print("BN bias: max: %f; mean: %f; min: %f; var: %f" %( m.bias.data.max(), m.bias.data.mean(), m.bias.data.min(), m.bias.data.var()))

    def showOrthInfo(self):
        ss = []
        for m_name, m in self.named_modules():
            if hasattr(m, 'showOrthInfo') and isinstance(m, self.myConv2d):
                s = m.showOrthInfo()
                s,_ = s.sort()
                ss.append(s.cpu().numpy())
        return ss
            # elif isinstance(m, nn.BatchNorm2d):
            #     print("BN Wei: ",m.weight.data,"  BN Bias: ",m.bias.data)

    def setmode(self, mode):
        for m_name, m in self.named_modules():
            if hasattr(m, 'setmode') and isinstance(m, self.myConv2d):
                m.setmode(mode)

    def project(self):
        for m_name, m in self.named_modules():
            if hasattr(m, 'project') and isinstance(m, self.myConv2d):
                m.project()

    def update_sigma(self,**kwargs):
        """
        Only for SN and UVR(mde 1 & 2)
        """
        for m_name, m in self.named_modules():
            if hasattr(m, 'update_sigma') and isinstance(m, self.myConv2d):
                m.update_sigma(**kwargs)

    def log_spectral(self):
        """
        Only for SN and UVR(mde 1 & 2)
        """
        log_s = 0
        for m_name, m in self.named_modules():
            if hasattr(m, 'log_spectral') and isinstance(m, self.myConv2d):
                log_s = log_s + m.log_spectral()
        return log_s

    def orth_penalty(self):
        penalty = 0
        for m_name, m in self.named_modules():
            if hasattr(m, 'orth_penalty') and isinstance(m, self.myConv2d):
                penalty = penalty + m.orth_penalty()
        return penalty

    def spectral_penalty(self):
        penalty = 0
        for m_name, m in self.named_modules():
            if hasattr(m, 'spectral_penalty') and isinstance(m, self.myConv2d):
                penalty = penalty + m.spectral_penalty()
        return penalty

resnet_G = Generator
resnet_D = Discriminator
