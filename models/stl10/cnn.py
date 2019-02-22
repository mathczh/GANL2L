import torch.nn as nn
import torch
from ..OrthConv import *
from ..SNConv import *
from ..SphereConv import *

__all__ = ['cnn_D','cnn_G','cnnv1_D','cnnv1_G','cnnv2_D','cnnv2_G','set_use_bias_cnn']

USE_BIAS = False

def set_use_bias_cnn(x):
    global USE_BIAS
    USE_BIAS = x

class CNN_D(nn.Module):

    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0, norm_type = 'none', loss_type='wgan', version = 1):
        super(CNN_D, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential()

        # input is nc x isize x isize
        if 'OR' in norm_type:
            if 'Mani' in norm_type:
                self.myConv2d = Orth_Plane_Mani_Conv2d
            else:
                self.myConv2d = Orth_Plane_Conv2d
        elif 'UVR' in norm_type:
            if 'Mani' in norm_type:
                self.myConv2d = Orth_UV_Mani_Conv2d
            else:
                self.myConv2d = Orth_UV_Conv2d
        elif 'WN' in norm_type:
            self.myConv2d = WN_Conv2d
        elif 'SN' in norm_type:
            self.myConv2d = SNConv2d
        elif 'Sphere' in norm_type:
            self.myConv2d = Sphere_Conv2d
        else:
            self.myConv2d = nn.Conv2d

        if version == 1:
            if 'OR' in norm_type:
                main.add_module('initial_4x4conv_{0}-{1}'.format(nc, ndf),
                                GroupOrthConv(self.myConv2d, nc, ndf, 4, 2, 1, bias=USE_BIAS))
            else:
                main.add_module('initial_4x4conv_{0}-{1}'.format(nc, ndf),
                                self.myConv2d(nc, ndf, 4, 2, 1, bias=USE_BIAS))
            if "BN" in norm_type:
                main.add_module('initial_{0}_4x4batchnorm'.format(ndf),
                                nn.BatchNorm2d(ndf))
            elif "LN" in norm_type:
                main.add_module('initial_{0}_4x4layernorm'.format(ndf),
                                nn.LayerNorm((ndf,isize//2,isize//2)))
            main.add_module('initial_4x4relu_{0}'.format(ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        if version == 2:
            # 3*3 block
            if 'OR' in norm_type:
                main.add_module('initial_3x3conv_{0}-{1}'.format(nc, ndf),
                                GroupOrthConv(self.myConv2d, nc, ndf, 3, 1, 1, bias=USE_BIAS))
            else:
                main.add_module('initial_3x3conv_{0}-{1}'.format(nc, ndf),
                                self.myConv2d(nc, ndf, 3, 1, 1, bias=USE_BIAS))
            if "BN" in norm_type:
                main.add_module('initial_{0}_3x3batchnorm'.format(ndf),
                                nn.BatchNorm2d(ndf))
            elif "LN" in norm_type:
                main.add_module('initial_{0}_3x3layernorm'.format(ndf),
                                nn.LayerNorm((ndf,isize,isize)))
            main.add_module('initial_3x3relu_{0}'.format(ndf),
                            nn.LeakyReLU(0.1, inplace=True))

            # 4*4 block
            main.add_module('initial_4x4conv_{0}-{1}'.format(ndf, ndf),
                            self.myConv2d(ndf, ndf, 4, 2, 1, bias=USE_BIAS))
            if "BN" in norm_type:
                main.add_module('initial_{0}_4x4batchnorm'.format(ndf),
                                nn.BatchNorm2d(ndf))
            elif "LN" in norm_type:
                main.add_module('initial_{0}_4x4layernorm'.format(ndf),
                                nn.LayerNorm((ndf,isize//2,isize//2)))
            main.add_module('initial_4x4relu_{0}'.format(ndf),
                            nn.LeakyReLU(0.1, inplace=True))
        csize, cndf = isize // 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}_conv'.format(t, cndf),
                            self.myConv2d(cndf, cndf, 3, 1, 1, bias=USE_BIAS))
            if "BN" in norm_type:
                main.add_module('extra-layers-{0}-{1}_batchnorm'.format(t, cndf),
                                nn.BatchNorm2d(cndf))
            elif "LN" in norm_type:
                main.add_module('extra-layers-{0}-{1}_layernorm'.format(t, cndf),
                                nn.LayerNorm((cndf,csize,csize)))
            main.add_module('extra-layers-{0}-{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 6:
            in_feat = cndf
            out_feat = cndf * 2

            ##############################################################
            # V1
            ##############################################################
            if version == 1:
                main.add_module('pyramid_{0}-{1}_4x4conv'.format(in_feat, out_feat),
                                self.myConv2d(in_feat, out_feat, 4, 2, 1, bias=USE_BIAS))
                if "BN" in norm_type:
                    main.add_module('pyramid_{0}_4x4batchnorm'.format(out_feat),
                                    nn.BatchNorm2d(out_feat))
                elif "LN" in norm_type:
                    main.add_module('pyramid_{0}_4x4layernorm'.format(out_feat),
                                    nn.LayerNorm((out_feat,csize//2,csize//2)))
                main.add_module('pyramid_{0}_4x4relu'.format(out_feat),
                                nn.LeakyReLU(0.2, inplace=True))

            ##############################################################
            # V2
            ##############################################################
            if version == 2:
                # 3*3 block
                main.add_module('pyramid_{0}-{1}_3x3conv'.format(in_feat, out_feat),
                                self.myConv2d(in_feat, out_feat, 3, 1, 1, bias=USE_BIAS))
                if "BN" in norm_type:
                    main.add_module('pyramid_{0}_3x3batchnorm'.format(out_feat),
                                    nn.BatchNorm2d(out_feat))
                elif "LN" in norm_type:
                    main.add_module('pyramid_{0}_3x3layernorm'.format(out_feat),
                                    nn.LayerNorm((out_feat,csize,csize)))
                main.add_module('pyramid_{0}_3x3relu'.format(out_feat),
                                nn.LeakyReLU(0.1, inplace=True))
                # 4*4 block
                main.add_module('pyramid_{0}-{1}_4x4conv'.format(out_feat, out_feat),
                                self.myConv2d(out_feat, out_feat, 4, 2, 1, bias=USE_BIAS))
                if "BN" in norm_type:
                    main.add_module('pyramid_{0}_4x4batchnorm'.format(out_feat),
                                    nn.BatchNorm2d(out_feat))
                elif "LN" in norm_type:
                    main.add_module('pyramid_{0}_4x4layernorm'.format(out_feat),
                                    nn.LayerNorm((out_feat,csize//2,csize//2)))
                main.add_module('pyramid_{0}_4x4relu'.format(out_feat),
                                nn.LeakyReLU(0.1, inplace=True))


            cndf = cndf * 2
            csize = csize // 2

        # state size. K x 4 x 4
        if version == 1:
            main.add_module('final_{0}-{1}_4x4conv'.format(cndf, 1),
                            self.myConv2d(cndf, 1, 6, 1, 0, bias=USE_BIAS))
        if version == 2:
            in_feat = cndf
            out_feat = cndf * 2
            # 3*3 block
            main.add_module('final_{0}-{1}_3x3conv'.format(in_feat, out_feat),
                            self.myConv2d(in_feat, out_feat, 3, 1, 1, bias=USE_BIAS))
            if "BN" in norm_type:
                main.add_module('final_{0}_3x3batchnorm'.format(out_feat),
                                nn.BatchNorm2d(out_feat))
            elif "LN" in norm_type:
                main.add_module('final_{0}_3x3layernorm'.format(out_feat),
                                nn.LayerNorm((out_feat,4,4)))
            main.add_module('final_{0}_3x3relu'.format(out_feat),
                            nn.LeakyReLU(0.1, inplace=True))
            main.add_module('final_{0}-{1}_4x4conv'.format(out_feat, 1),
                            self.myConv2d(out_feat, 1, 6, 1, 0, bias=USE_BIAS))
            cndf = cndf * 2

        if loss_type == "dcgan":
            main.add_module('final_sigmoid', nn.Sigmoid())
        self.main = main

        for m_name, m in self.named_modules():
            print(m_name+': '+m.__class__.__name__)
            if isinstance(m, nn.Conv2d): # Special Convlayer has their own ini
                m.weight.data.normal_(0,0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.squeeze()

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




def cnn_D(isize, nc, ndf, ngpu, n_extra_layers=0, norm_type = 'none', loss_type='wgan'):
    model = CNN_D(isize, nc, ndf, ngpu, n_extra_layers, norm_type, loss_type, version = 1)
    return model

cnnv1_D = cnn_D

def cnnv2_D(isize, nc, ndf, ngpu, n_extra_layers=0, norm_type = 'none', loss_type='wgan'):
    model = CNN_D(isize, nc, ndf, ngpu, n_extra_layers, norm_type, loss_type, version = 2)
    return model

class CNN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(CNN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 6, 1, 0, bias=USE_BIAS))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=USE_BIAS))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=USE_BIAS))
            main.add_module('extra-layers-{0}-{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=USE_BIAS))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

        for m_name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d): # Special Convlayer has their own ini
                m.weight.data.normal_(0,0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


def cnn_G(isize, nz, nc, ngf, ngpu, n_extra_layers=0):
    model = CNN_G(isize, nz, nc, ngf, ngpu, n_extra_layers)
    return model

cnnv1_G = cnn_G


class CNNv2_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(CNNv2_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 6
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        cngf = cngf * 2
        tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 6, 1, 0))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 6, cngf
        while csize <= isize//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 3, 1, 1))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

        for m_name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d): # Special Convlayer has their own ini
                m.weight.data.normal_(0,0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

def cnnv2_G(isize, nz, nc, ngf, ngpu):
    model = CNNv2_G(isize, nz, nc, ngf, ngpu)
    return model
