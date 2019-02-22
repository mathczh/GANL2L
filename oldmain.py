from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
import torch.autograd as autograd
from torch.autograd import Variable

import argparse
import numpy as np
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
from utils import inception_score
from utils import CustomImgDataset
# from is_utils import get_inception_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--arch', required=True, help='cnn | resnet (TODO)')
parser.add_argument('--loss', default='wgan', help='wgan | dcgan')
parser.add_argument('--dataroot', default='~/dataset/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=256, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default= 0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default= 0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--norm_type', default='', help =
'''
Can be 'OR':
    Orthonormal Regularization \lambda ||WtW-I||_2^2
Can be 'OR+Mani'
Can be 'UVR':
    Decompose W into UDV, and add Orthonormal Regularization on U&V \lambda ||UtU-I||_2^2 + ||VtV-I||_2^2
Can be 'UVR+Mani':
Can be 'WN':
    Weight Normalization
Can be 'Sphere':
    Sphere Normalization

Can be 'BN':
    Batch Normalization
Can be 'LN':
    Layer Normalization

Can be 'GP':
    Gradient Penalty

Can be 'WC':
    Weight Clipping param norm to c (Wasserstein distance and Lipschitz continuity)

Or you can use like 'OR+BN+WC' or 'OR+BN'
''')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--gpwei', type=float, default=10)
parser.add_argument('--orthwei', type=float, default=1000)
parser.add_argument('--orscale', type=float, default=1) #TODO
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--addinfo', default="", help='additional information show up in path')
parser.add_argument('--opt', default='adam', help='adam | rmsprop')
parser.add_argument('--use_proj', action='store_true', help='apply projection after Optimization')
parser.add_argument('--show_sv_info', action='store_true', help='apply projection after Optimization')
opt = parser.parse_args()

if opt.experiment is None:
    opt.experiment = os.path.dirname(os.path.abspath(__file__))+"/"+opt.dataset+'_'+opt.loss+'_'+opt.norm_type+('_PROJ' if opt.use_proj else '')+'_'+opt.arch
os.system('mkdir {0}'.format(opt.experiment))
sys.stdout = Logger(opt.experiment+"/log.txt","w", sys.stdout)
print(opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if opt.cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
FloatTensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if opt.cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if opt.cuda else torch.ByteTensor
Tensor = FloatTensor


if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

### set network
if opt.dataset.startswith("cifar"):
    import models.cifar as models
    netD = models.__dict__[opt.arch+'_D'](opt.imageSize, nc, ngf, ngpu, n_extra_layers, opt.norm_type, opt.loss)
    netG = models.__dict__[opt.arch+'_G'](opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)

if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.opt == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
elif opt.opt == 'rmsprop':
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

def calc_gradient_penalty(netD, x, g):
    assert x.size() == g.size()
    a = torch.rand(x.size(0), 1)
    a = a.cuda() if opt.cuda else a
    a = a\
        .expand(x.size(0), x.nelement()//x.size(0))\
        .contiguous()\
        .view(
            x.size(0),
            nc,
            opt.imageSize,
            opt.imageSize
        )
    interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
    c = netD(interpolated)
    gradients = autograd.grad(
        outputs=c, inputs=interpolated,
        grad_outputs=(
            torch.ones(c.size()).cuda() if opt.cuda else
            torch.ones(c.size())
        ),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return opt.gpwei * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Start Training
print("=================================================================")
print("=======================Start  Training===========================")
print("=================================================================")
gen_iterations = 0
IS_array = []
bestIS = 0
if opt.loss == "dcgan":
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ####################################################
        # ----- train model_D -----
        # WGAN: maximize D(x) - D(G(z))
        # DCGAN: maximize log(D(x)) + log(1 - D(G(z)))
        ####################################################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        Diters = opt.Diters
        # if gen_iterations < 25 or gen_iterations % 500 == 0:
        #     Diters = 100
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1

            # clamp parameters to a cube
            if 'WC' in opt.norm_type:
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            if 'SN' in opt.norm_type:
                netD.update_sigma()

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad()
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            real_inputv = Variable(input)
            output = netD(real_inputv)

            if opt.loss == "wgan":
                errD_real = -output.mean()
            elif opt.loss == "dcgan":
                label = FloatTensor(batch_size,)
                label.fill_(real_label)
                labelv = Variable(label)
                errD_real = criterion(output, labelv)
            errD_real.backward()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev).data)
            fake_inputv = fake
            output = netD(fake_inputv)
            if opt.loss == "wgan":
                errD_fake = output.mean()
            elif opt.loss == "dcgan":
                label.fill_(fake_label)
                labelv = Variable(label)
                errD_fake = criterion(output, labelv)
            errD_fake.backward()
            errD = errD_real + errD_fake
            Distribution_D = errD

            if 'GP' in opt.norm_type:
                gradient_penalty = calc_gradient_penalty(netD, real_inputv, fake)
                gradient_penalty.backward()
                errD = errD + gradient_penalty

            if 'OR' in opt.norm_type or 'UVR' in opt.norm_type:
                orth_wei = opt.orthwei * (0.01**(epoch/opt.niter))
                orth_penalty = netD.orth_penalty()*orth_wei
                orth_penalty.backward()
                errD = errD + orth_penalty

            optimizerD.step()
            if opt.use_proj:
                netD.project()


        ####################################################
        # ------ train model_G -------
        # WGAN: maximize D(G(z))
        # DCGAN; maximize log(D(G(z)))
        # train model_D more: because the better model_D is,
        # the better model_G will be
        ####################################################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        output = netD(fake)
        if opt.loss == "wgan":
            errG = -output.mean()
        elif opt.loss == "dcgan":
            label = FloatTensor(opt.batchSize)
            label.fill_(real_label) # fake labels are real for generator cost logd trick
            labelv = Variable(label)
            errG = criterion(output, labelv)
        errG.backward()
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Distribution_D: %f Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataloader), gen_iterations, Distribution_D,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        if gen_iterations % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            fake = netG(Variable(fixed_noise, volatile=True))
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    ###
    if(opt.show_sv_info):
        print ("Print Singular Value Information...")
        netD.showOrthInfo()

    ### Calculate inception_score
    # Generate 5000 data
    noisev = Variable(FloatTensor(5000,nz,1,1).normal_(0,1), volatile=True)
    netG.eval()
    print ("Calculating Inception Score...")
    fask_imgs = netG(noisev)
    IS = inception_score(CustomImgDataset(fask_imgs.data), batch_size=opt.batchSize, cuda=opt.cuda, splits=10)
    # fask_imgs = np.transpose(netG(noisev).data.cpu().numpy(),(0,2,3,1))
    # fask_imgs = (fask_imgs*0.5+0.5)*255
    # IS = get_inception_score(fask_imgs)
    IS_array.append(IS)
    print(IS)
    if bestIS < IS[0]:
        bestIS = IS[0]
        fask_imgs = np.transpose(netG(noisev).data.cpu().numpy(),(0,2,3,1))
        fask_imgs = (fask_imgs*0.5+0.5)*255
        np.save(opt.experiment+"/fakeimgs_best.npy",fask_imgs)

    if epoch%10 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

noisev = Variable(FloatTensor(5000,nz,1,1).normal_(0,1), volatile=True)
netG.eval()
fask_imgs = np.transpose(netG(noisev).data.cpu().numpy(),(0,2,3,1))
fask_imgs = (fask_imgs*0.5+0.5)*255
np.save(opt.experiment+"/fakeimgs_final.npy",fask_imgs)
shutil.copy2('./tfIS.py', opt.experiment)
shutil.copy2('./is_utils.py', opt.experiment)


IS_array = np.array(IS_array)
plt.plot(range(len(IS_array[:,0])), IS_array[:,0], 'r-', label = 'Inception Score')
plt.legend()
plt.savefig(opt.experiment+'/inception_score.pdf', bbox_inches='tight',format="pdf", dpi = 300)
plt.close()

np.save(opt.experiment+"/inception_score.npy",{"IS_array":IS_array})

noise = Variable(FloatTensor(64,nz,1,1).normal_(0,1))
fake_u=netG(noise)
imgs = make_grid(fake_u.data*0.5+0.5).cpu() # CHW
plt.figure(figsize=(5,5))
plt.imshow(imgs.permute(1,2,0).numpy()) # HWC
plt.savefig(opt.experiment+'/final_figures.pdf', bbox_inches='tight',format="pdf", dpi = 300)
plt.close()
