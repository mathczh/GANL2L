# OrthGAN
Pytorch implementation of OrthGAN

## Our method
We propose to use shpere Convolution Structure and annealing Orthonormal Regularization to train GAN.

## Problem List
- How to control spectral norm for each layer?
- BN?
- dist of eigen values?

## TODO List
**Functionality**
- [x] Show images
- [x] Show Loss
- [x] Show inception score
- [ ] Show FID
- [x] Use WGAN Loss
- [x] Use DCGAN Loss
- [x] Use Hinge Loss

**Normalization Style**
- [x] [Weight clipping (WC)](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf)
- [x] [Batch-normalization (BN)](https://arxiv.org/pdf/1502.03167.pdf)
- [x] [Layer normalization (LN)](https://arxiv.org/pdf/1607.06450.pdf)
- [x] [Weight normalization (WN)](https://arxiv.org/pdf/1602.07868.pdf)
- [x] [Deep Hyperspherical Learning(Sphere)](https://arxiv.org/pdf/1711.03189.pdf)
- [x] [Orthonormal regularization (OR)](https://arxiv.org/pdf/1609.07093.pdf)
- [x] [Spectral normalization (SN)](https://openreview.net/pdf?id=B1QRgziT-)
- [x] UV
- [x] [Gradient Penalty WGAN-GP]((https://arxiv.org/pdf/1704.00028.pdf))

*Note: we excluded the multiplier parameter γ in the weight normalization method, as well as in batch normalization and layer normalization method. This was done in order to prevent the methods from overtly violating the Lipschitz condition.*

**Add Network**
- [x] CNN
- [ ] ResNet

**More Dataset**
- [ ] Add LSUN dataset
- [ ] Add STL-10 dataset
- [ ] Add imagenet

# CNN Structure for Cifar-10
**Generator**  
noise(input): $z \in R^{128} \sim N(0,I)$  
```python
width=64

nn.ConvTranspose2d(noise_dim,width*8,4,1,0,bias=False),
nn.BatchNorm2d(width*8),
nn.ReLU(True),

nn.ConvTranspose2d(width*8,width*4,4,2,1,bias=False),
nn.BatchNorm2d(width*4),
nn.ReLU(True),

nn.ConvTranspose2d(width*4,width*2,4,2,1,bias=False),
nn.BatchNorm2d(width*2),
nn.ReLU(True),

nn.ConvTranspose2d(width*2,width,4,2,1,bias=False),
nn.BatchNorm2d(width),
nn.ReLU(True),

nn.ConvTranspose2d(width,opt.nc,4,2,1,bias=False),
nn.Tanh()
```


**Discriminator**  
noise(input): $x \in R^{M \times M \times 3}$  
```python
width = 64

# input is 3 x 32 x 32
nn.Conv2d(3,width,4,2,1,bias=False),
nn.LeakyReLU(0.2,inplace=True),
# width x 16 x 16
nn.Conv2d(width,width*2,4,2,1,bias=False),
nn.BatchNorm2d(width*2),
nn.LeakyReLU(0.2,inplace=True),
# 2width x 8 x 8
nn.Conv2d(width*2,width*4,4,2,1,bias=False),
nn.BatchNorm2d(width*4),
nn.LeakyReLU(0.2,inplace=True),
# 4width x 4 x 4
nn.Conv2d(width*4,width*8,4,2,1,bias=False),
nn.BatchNorm2d(width*8),
nn.LeakyReLU(0.2,inplace=True),
# linear layer
nn.Conv2d(width*8,1,4,1,0,bias=False),
```

**Optimization Setting**
```python
##DCGAN
RANDOMSEED = None
LEARNING_RATE_D = 0.0001
LEARNING_RATE_G = 0.0001
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
WEIGHT_DECAY = 0
EPOCHES = 60
LRDECAY_EPOCHES = None
LRDECAY_SCHEDULE = [35]
LRDECAYRATE = 0.1
BATCH_SIZE = 64
IMG_SIZE = 64

NOISE_DIM = 128 # noise dimension
NDIS = 5 # the number of updates of the discriminator per one update of the generator
```

## ResNet Structure for Cifar-10

## ResNet Structure for STL-10

## Result

| Dataset | Network | Loss | Regularization | Inception Score |
| --- | --- | --- | --- | :---: |
| Cifar10 | real (tf) |       |    |   |
| Cifar10 | real (pytorch) |       |    | (9.866939554846898, 0.18634018911911107)  |
| Cifar10 | CNN | wgan | WC | Failed |
| Cifar10 | CNN | wgan | BN | (3.7321552971233842, 0.0699739890119141) (Loss explode) |
| Cifar10 | CNN | wgan | WC+BN | (4.1719824477539005, 0.1150727362245363) |
| Cifar10 | CNN | wgan | WN |  |
| Cifar10 | CNN | wgan | WN+BN |  |
| Cifar10 | CNN | wgan | Sphere |  |
| Cifar10 | CNN | wgan | Sphere+BN |  |
| Cifar10 | CNN | wgan | LN |  |
| Cifar10 | CNN | wgan | SN |  |
| Cifar10 | CNN | wgan | OR |  |
| Cifar10 | CNN | wgan | OR+BN |  |
| Cifar10 | CNN | wgan | UVR |  |
| Cifar10 | CNN | wgan | UVR+BN |  |
| Cifar10 | CNN | dcgan | WC | (3.8075969665634886, 0.16619519095483568) |
| Cifar10 | CNN | dcgan | BN | (4.349229302072816, 0.17452000129824255) |
| Cifar10 | CNN | dcgan | WC+BN | (4.27193365127717, 0.12600791315663723) |
| Cifar10 | CNN | dcgan | WN |  |
| Cifar10 | CNN | dcgan | WN+BN |  |
| Cifar10 | CNN | dcgan | Sphere |  |
| Cifar10 | CNN | dcgan | Sphere+BN |  |
| Cifar10 | CNN | dcgan | LN |  |
| Cifar10 | CNN | dcgan | SN |  |
| Cifar10 | CNN | dcgan | OR |  |
| Cifar10 | CNN | dcgan | OR+BN |  |
| Cifar10 | CNN | dcgan | UVR |  |
| Cifar10 | CNN | dcgan | UVR+BN |  |
| Cifar10 | CNN | hinge | WC |  |
| Cifar10 | CNN | hinge | BN |  |
| Cifar10 | CNN | hinge | WC+BN |  |
| Cifar10 | CNN | hinge | WN |  |
| Cifar10 | CNN | hinge | WN+BN |  |
| Cifar10 | CNN | hinge | Sphere |  |
| Cifar10 | CNN | hinge | Sphere+BN |  |
| Cifar10 | CNN | hinge | LN |  |
| Cifar10 | CNN | hinge | SN |  |
| Cifar10 | CNN | hinge | OR |  |
| Cifar10 | CNN | hinge | OR+BN |  |
| Cifar10 | CNN | hinge | UVR |  |
| Cifar10 | CNN | hinge | UVR+BN |  |

## Reference
[Spectral Normalization for Generative Adversarial Networks](https://openreview.net/forum?id=B1QRgziT-)  
[cGANs with Projection Discriminator](https://openreview.net/forum?id=ByS1VpgRZ)  
[Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)  
[Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)  (WN)  
[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

**Theory**  
[Generalization and Equilibrium in Generative Adversarial Nets (GANs)](https://arxiv.org/abs/1703.00573)  
[Spectrally-normalized margin bounds for neural networks](https://arxiv.org/abs/1706.08498)  
