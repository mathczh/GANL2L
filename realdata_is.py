import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from utils import inception_score
from utils import CustomImgDataset
#from is_utils import get_inception_score

from torchvision.models.inception import inception_v3
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


cifar = dset.CIFAR10(root="~/dataset/", download=True,
                         transform=transforms.Compose([
                             transforms.Scale(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
)

print ("Calculating Inception Score...")
#print(get_inception_score(cifar.train_data))

IgnoreLabelDataset(cifar)
print (inception_score(IgnoreLabelDataset(cifar), cuda=True, resize=True, splits=10))
