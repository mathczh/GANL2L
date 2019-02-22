'''
1. remove sigmoid in the last layer of discriminator(classification -> regression)
2. no log Loss (Wasserstein distance)
3. No momentum-based optimizer, use SGD instead
'''

DEBUGMODE = False

# import packages
import argparse
import numpy as np
from is_utils import get_inception_score
# preheat tensorflow to avoid memory problem
if(not DEBUGMODE):
    print("Preheating Tensorflow")
    print(get_inception_score(100*np.ones([10,32,32,3]),splits=10))
import random
import os
import sys
import shutil
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import Logger
#from utils import inception_score
#from utils import CustomImgDataset

# import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import RMSprop
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid

from models.projsgd import ProjSGD

# Define Parameters
ARCH = "cnnv1" # cnn | cnnv2
DATASET = "cifar10"
DATAPATH = "~/dataset/"
ADDITIONAL_INFO = ""
RANDOMSEED = None
LEARNING_RATE_D = 0.0001
LEARNING_RATE_G = 0.0001
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
WEIGHT_DECAY = 0
EPOCHES = 256
LRDECAY_EPOCHES = None
LRDECAY_SCHEDULE = [35]
LRDECAYRATE = 0.1
BATCH_SIZE = 64
IMG_SIZE = 32

NOISE_DIM = 128 # noise dimension
NDIS = 1 # the number of updates of the discriminator per one update of the generator

USE_PRETRAINED = False  # only worked for imagenet
RESUME_PATH = ''
START_EPOCH = 0
LOAD_WORKER = 4

IMG_SAVE_EPOCH = 1
SHOW_SV_INFO = False
SHOW_BN_INFO = False

USE_BIAS = False

# Loss Style
LOSS_TYPE = 'dcgan'
'''
Can be 'wgan':
    D: maximize D(x) - D(G(z))
    G: maximize D(G(z))
Can be 'dcgan':
    apply sigmoid on D
    D: maximize log(D(x)) + log(1 - D(G(z)))
    G: maximize log(D(G(z)))
Can be 'hinge':
    apply sigmoid on D
    D: maximize min[0,-1+D(x)] + min[0,-1-D(G(z))]
    G: maximize D(G(z))
'''

# Normalization Style
NORM_TYPE = 'UVR'
'''
You can use 'OR+BN+WC' or 'OR+BN'

Can be 'OR':
    Orthonormal Regularization \lambda ||WtW-I||_2^2
Can be 'UVR':
    Decompose W into UDV, and add Orthonormal Regularization on U&V \lambda ||UtU-I||_2^2 + ||VtV-I||_2^2
    Mode 1: divided by max
    Mode 2: truncate by 1
    Mode 3: penalize sum of all log max spectral
    Mode 4: penalize prod max spectral

Can be 'WN':
    Weight Normalization
Can be 'Sphere':
    Sphere Normalization
Can be 'GP':
    Gradient Penalty

Can be 'BN':
    Batch Normalization
Can be 'LN':
    Layer Normalization

Can be 'WC':
    Weight Clipping param norm to c (Wasserstein distance and Lipschitz continuity)
'''
CLAMP_BAR = 0.01 # weight clip bar
GP_WEI = 10 # Gradient Penalty Weight
ORTH_WEI = 1 # for OR & UVR
ANNEAL_ORTHWEI = False ## totally not useful
FINAL_ORTHWEI_RATIO = 0.01
UVR_MODE = 1
'''
Mode 1: divided by max
Mode 2: truncate by 1
Mode 3: penalize sum of all log max spectral
Mode 4: clip to [0.5,1]
'''
SPECT_WEI = .001 # for UVR Mode 3
ANNEAL_SPECTWEI = False ## totally not useful
FINAL_SPECTWEI_RATIO = 0.01

# The following parameter only control the special layer for special convlayer
USE_PROJ = False

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('GPU IS AVAILABLE TO USE')
    cudnn.benchmark = True
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


stdout_backup = sys.stdout

def normalize_img(t,tmin = -1, tmax = 1):
    if tmin is None:
        tmin = float(t.min())
    if tmax is None:
        tmax = float(t.max())
    return (t-tmin)/(tmax - tmin + 1e-20)

def calc_gradient_penalty(netD, x, g):
    assert x.size() == g.size()
    a = torch.rand(x.size(0), 1)
    a = a.cuda() if USE_CUDA else a
    a = a\
        .expand(x.size(0), x.nelement()//x.size(0))\
        .contiguous()\
        .view(
            x.size(0),
            3,
            IMG_SIZE,
            IMG_SIZE
        )
    interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
    c = netD(interpolated).mean()
    gradients = autograd.grad(
        outputs=c, inputs=interpolated,
        grad_outputs=(
            torch.ones(c.size()).cuda() if USE_CUDA else
            torch.ones(c.size())
        ),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def main():

    if DATASET.startswith("cifar") or DATASET.startswith("lsun"):
        import models.cifar as models
    if DATASET.startswith("stl10"):
        import models.stl10 as models
    if DATASET.startswith("imagenet"):
        import models.imagenet as models

    models.set_use_bias_cnn(USE_BIAS)
    models.set_use_bias_res(USE_BIAS)

    # make dictionary to store model
    ROOTPATH = os.path.dirname(os.path.abspath(__file__))+"/"+DATASET+'_'+LOSS_TYPE+'_'+NORM_TYPE+('_PROJ' if USE_PROJ else '')+'_'+ARCH+ADDITIONAL_INFO
    if not os.path.exists(ROOTPATH):
        os.mkdir(ROOTPATH)

    sys.stdout = Logger(ROOTPATH+"/log.txt","w", stdout_backup)

    print('ROOTPATH: '+ROOTPATH)
    # Random seed
    global RANDOMSEED
    if RANDOMSEED == None:
        RANDOMSEED = random.randint(1, 10000)
    random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOMSEED)
    print("use random seed: "+str(RANDOMSEED))

    # setup data loader
    if DATASET.startswith('cifar'):
        data_transform = torchvision.transforms.Compose(
            [   transforms.Resize(IMG_SIZE),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        if DATASET == 'cifar10':
            dataset = dset.CIFAR10(DATAPATH, transform=data_transform, download=False)
            num_classes = 10
        elif DATASET == 'cifar100':
            dataset = dset.CIFAR100(DATAPATH, transform=data_transform, download=False)
            num_classes = 100
    elif DATASET == 'lsun':
        dataset = dset.LSUN(DATAPATH, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(IMG_SIZE),
                                transforms.CenterCrop(IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif DATASET == 'stl10':
        dataset = dset.STL10(DATAPATH, split='train',
                            transform=transforms.Compose([
                                transforms.Resize(IMG_SIZE),
                                transforms.CenterCrop(IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]), download=False)
    elif opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        raise RuntimeError(DATASET+'not implemented yet')
    print('Number of real samples: ', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=LOAD_WORKER)

    # set up model
    if "cnn" in ARCH:
        model_D = models.__dict__[ARCH+'_D'](IMG_SIZE,3,64,1,norm_type = NORM_TYPE,loss_type = LOSS_TYPE).to(DEVICE)
        model_G = models.__dict__[ARCH+'_G'](IMG_SIZE,NOISE_DIM,3,64,1).to(DEVICE)
    if "resnet" in ARCH:
        model_D = models.__dict__[ARCH+'_D'](IMG_SIZE,3,128,1,norm_type = NORM_TYPE,loss_type = LOSS_TYPE).to(DEVICE)
        model_G = models.__dict__[ARCH+'_G'](IMG_SIZE,NOISE_DIM,3,256,1).to(DEVICE)
    # model_D = models.__dict__[ARCH+'_D'](3, 64).to(DEVICE)
    if "UVR" in NORM_TYPE:
        model_D.setmode(UVR_MODE)

    print(model_G)
    print(model_D)

    print('    Total params (D): %.2fM' % (sum(p.numel() for p in model_D.parameters())/1000000.0))
    print('    Total params (G): %.2fM' % (sum(p.numel() for p in model_G.parameters())/1000000.0))

    # define optimizer
    optimizer_D = optim.Adam(model_D.parameters(), lr = LEARNING_RATE_D, betas=(ADAM_BETA1, ADAM_BETA2))
    optimizer_G = optim.Adam(model_G.parameters(), lr = LEARNING_RATE_G, betas=(ADAM_BETA1, ADAM_BETA2))

    # check if we can load checkpoint
    global START_EPOCH
    if RESUME_PATH:
        if os.path.isfile(RESUME_PATH):
            print("=> loading checkpoint '{}'".format(RESUME_PATH))
            checkpoint = torch.load(RESUME_PATH)
            START_EPOCH = checkpoint['epoch']
            model_D.load_state_dict(checkpoint['state_dict_D'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            model_G.load_state_dict(checkpoint['state_dict_G'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(RESUME_PATH, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(RESUME_PATH))

    # Start Training
    print("=================================================================")
    print("=======================Start  Training===========================")
    print("=================================================================")

    fix_noise = torch.randn(BATCH_SIZE,NOISE_DIM, 1, 1, device=DEVICE)

    IS_array = []
    bestIS = 0

    for epoch in range(START_EPOCH, EPOCHES):
        # adjust_learning_rate(optimizer_D, epoch,LEARNING_RATE_D,LRDECAY_SCHEDULE)
        # adjust_learning_rate(optimizer_G, epoch,LEARNING_RATE_G,LRDECAY_SCHEDULE)

        train(dataloader, model_D, model_G, optimizer_D, optimizer_G, epoch)

        if epoch%IMG_SAVE_EPOCH == 0:
            fake_u=model_G(fix_noise)
            # imgs = make_grid(fake_u.data*0.5+0.5) # CHW
            # vutils.save_image(fake_u.data,
            #         '%s/fake_samples_epoch_%03d.png' % (ROOTPATH, epoch),
            #         normalize=True)
            print("fake_u min: ", fake_u.min().item())
            print("fake_u max: ", fake_u.max().item())
            print("fake_u var: ", fake_u.var().item())
            imgs = make_grid(normalize_img(fake_u.data)).cpu() # CHW
            plt.figure(figsize=(5,5))
            plt.imshow(imgs.permute(1,2,0).numpy()) # HWC
            plt.savefig('%s/fake_samples_epoch_%03d.png' % (ROOTPATH, epoch),
                        bbox_inches='tight',format="png", dpi = 300)
            plt.close()


        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': ARCH,
            'state_dict_D': model_D.state_dict(),
            'optimizer_D' : optimizer_D.state_dict(),
            'state_dict_G': model_G.state_dict(),
            'optimizer_G' : optimizer_G.state_dict(),
        }, epoch, savepath = ROOTPATH)

        ###
        if(SHOW_SV_INFO):
            print ("Print Singular Value Information...")
            ss = model_D.showOrthInfo()
            layerid = -1
            for s in ss:
                layerid = layerid+1
                plt.plot(np.array(range(len(s)))/len(s), s, label = '%d-th layer'%(layerid))
            plt.legend()
            plt.savefig(ROOTPATH+'/sv_dist.pdf', bbox_inches='tight',format="pdf", dpi = 300)
            plt.close()
        if(SHOW_BN_INFO):
            print ("Print BatchNorm Information...")
            model_D.showBNInfo()
        ### Calculate inception_score
        # Generate 1000 data
        with torch.no_grad():
            print ("Generate Fake Images...")
            fake_imgs = None
            for i in range(10):
                noise = Variable(FloatTensor(100,NOISE_DIM,1,1).normal_(0,1), requires_grad=False)
                model_G.eval()
                fake_imgs_batch = model_G(noise)
                fake_imgs_batch = np.transpose(fake_imgs_batch.data.cpu().numpy(),(0,2,3,1))
                fake_imgs_batch = (normalize_img(fake_imgs_batch))*255
                fake_imgs = np.concatenate((fake_imgs_batch, fake_imgs), axis=0) if fake_imgs is not None else fake_imgs_batch
            print ("Calculating Inception Score...")
            # normalize imgs
            IS = (0,0)
            if(not DEBUGMODE):
                IS = get_inception_score(fake_imgs,splits=10)
                IS_array.append(IS)
                print(IS)
                if bestIS < IS[0]:
                    bestIS = IS[0]
                    save_best_checkpoint({
                        'epoch': epoch + 1,
                        'arch': ARCH,
                        'state_dict_D': model_D.state_dict(),
                        'optimizer_D' : optimizer_D.state_dict(),
                        'state_dict_G': model_G.state_dict(),
                        'optimizer_G' : optimizer_G.state_dict(),
                    }, epoch, savepath = ROOTPATH)

                _IS_array = np.array(IS_array)
                plt.plot(range(len(_IS_array[:,0])), _IS_array[:,0], 'r-', label = 'Inception Score')
                plt.legend()
                plt.savefig(ROOTPATH+'/inception_score.pdf', bbox_inches='tight',format="pdf", dpi = 300)
                plt.close()

                np.save(ROOTPATH+"/inception_score.npy",{"IS_array":_IS_array})
            else:
                np.save('%s/fake_imgs_%03d.npy' % (ROOTPATH, epoch),fake_imgs)


    if(not DEBUGMODE):
        IS_array = np.array(IS_array)
        plt.plot(range(len(IS_array[:,0])), IS_array[:,0], 'r-', label = 'Inception Score')
        plt.legend()
        plt.savefig(ROOTPATH+'/inception_score.pdf', bbox_inches='tight',format="pdf", dpi = 300)
        plt.close()

        np.save(ROOTPATH+"/inception_score.npy",{"IS_array":IS_array})

    with torch.no_grad():
        noise = Variable(FloatTensor(64,NOISE_DIM,1,1).normal_(0,1))
        fake_u=model_G(noise)
        imgs = make_grid(normalize_img(fake_u)).cpu() # CHW
        plt.figure(figsize=(5,5))
        plt.imshow(imgs.permute(1,2,0).numpy()) # HWC
        plt.savefig(ROOTPATH+'/final_figures.pdf', bbox_inches='tight',format="pdf", dpi = 300)
        plt.close()

    # Svae Singluar Value
    if(SHOW_SV_INFO):
        np.save(ROOTPATH+'/singluar_value.npy',{"ss":ss})

def train(dataloader, model_D, model_G, optimizer_D, optimizer_G, epoch):
    torch.set_grad_enabled(True)
    model_D.train()
    model_G.train()
    relu = nn.ReLU()
    criterion = nn.BCELoss()

    orth_penalty = None
    spectral_penalty = None

    # train for one epoch
    for ii, data in enumerate(dataloader,0):
        # modification: clip param for discriminator
        if 'WC' in NORM_TYPE:
            for parm in model_D.parameters():
                    parm.data.clamp_(-CLAMP_BAR,CLAMP_BAR)

        ####################################################
        # ----- train model_D -----
        # DCGAN:  maximize log(D(x)) + log(1 - D(G(z)))
        # WGAN:   maximize D(x) - D(G(z))
        # Hinge:  maximize min[0,-1+D(x)] + min[0,-1-D(G(z))]
        ####################################################
        model_D.zero_grad()
        ## train model_D with real img
        real =data[0].to(DEVICE)
        batch_size = real.size(0)

        output = model_D(real)
        if LOSS_TYPE == "dcgan":
            label = torch.full((batch_size,), 1, device=DEVICE)
            errD_real = criterion(output, label)
        if LOSS_TYPE == "wgan":
            errD_real = -output
        errD_real.backward()

        ## train model_D with fake img
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=DEVICE)
        fake_pic = model_G(noise)
        output2 = model_D(fake_pic.detach())
        if LOSS_TYPE == "dcgan":
            label.fill_(0)
            errD_fake = criterion(output2, label)
        if LOSS_TYPE == "wgan":
            errD_real = output2
        errD_fake.backward()

        errD = errD_real+errD_fake
        Distribution_D = errD

        if 'GP' in NORM_TYPE:
            gradient_penalty = GP_WEI * calc_gradient_penalty(model_D, inputv, fake_pic.detach())
            gradient_penalty.backward()
            errD = errD + gradient_penalty

        if 'OR' in NORM_TYPE or 'UVR' in NORM_TYPE:
            if ANNEAL_ORTHWEI:
                orth_wei = ORTH_WEI * (FINAL_ORTHWEI_RATIO**(epoch/EPOCHES))
            else:
                orth_wei = ORTH_WEI
            orth_penalty = model_D.orth_penalty()*orth_wei
            orth_penalty.backward()
            errD = errD + orth_penalty

        if 'UVR' in NORM_TYPE and UVR_MODE in (3,5,6,7,8,9,10):
            spectral_penalty = 0
            if ANNEAL_SPECTWEI:
                spectral_wei = SPECT_WEI * (FINAL_SPECTWEI_RATIO**(epoch/EPOCHES))
            else:
                spectral_wei = SPECT_WEI
            if UVR_MODE in (3,5):
                spectral_penalty += relu(model_D.log_spectral())*spectral_wei
            elif UVR_MODE in (5,6,7,8,9,10):
                spectral_penalty += model_D.spectral_penalty()*spectral_wei
            spectral_penalty.backward()
            errD = errD + spectral_penalty


        optimizer_D.step()

        if USE_PROJ:
            model_D.project()

        ####################################################
        # ------ train model_G -------
        # DCGAN: maximize log(D(G(z)))
        # WGAN: maximize D(G(z))
        # Hinge: maximize D(G(z))
        # train model_D more: because the better model_D is,
        # the better model_G will be
        ####################################################
        if (ii+1)%NDIS ==0:
            model_G.zero_grad()
            model_D.zero_grad()
            if False: # resample fake_pic
                noise.resize_(batch_size, NOISE_DIM, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake_pic = model_G(noisev)
            output3 = model_D(fake_pic)
            if LOSS_TYPE == "dcgan":
                label.fill_(1)
                errG = criterion(output3, label)
            if LOSS_TYPE == "wgan":
                errG = -output3
            errG.backward()

            optimizer_G.step()

            print_info = '[%d/%d][%d/%d] Distribution_D: %f Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'\
                % (epoch, EPOCHES, ii, len(dataloader), Distribution_D,
                errD.data.item(), errG.data.item(), errD_real.data.item(), errD_fake.data.item())
            if orth_penalty is not None:
                print_info += ' orth_penalty: %f' % (orth_penalty.data.item())
            if spectral_penalty is not None:
                print_info += ' spectral_penalty: %f' % (spectral_penalty.data.item())
            print(print_info)

def save_checkpoint(state, epoch, savepath='./'):
    torch.save(state, savepath+'/checkpoint.pth.tar')

def save_best_checkpoint(state, epoch, savepath='./'):
    torch.save(state, savepath+'/best_checkpoint.pth.tar')

def adjust_learning_rate(optimizer, epoch, inilr, lr_schedule = []):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_schedule == []:
        lr = inilr * (LRDECAYRATE ** (epoch // LRDECAY_EPOCHES))
    else:
        lr = inilr
        for schedule_epo in lr_schedule:
            if epoch >= schedule_epo:
                lr = lr * LRDECAYRATE
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser(description='Write your test code in a saperate py file.')
parser.add_argument('--filepath', '-f', metavar='filepath', required=True,
                    help='the path of the test file, which should be a py file')
args = parser.parse_args()

if __name__ == '__main__':
    exec(open(args.filepath).read())
