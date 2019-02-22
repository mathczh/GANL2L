import sys
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy

from torchvision.models.inception import inception_v3
#import pretrainedmodels

class Logger(object):
    def __init__(self, filepath = "./log.txt", mode = "w", stdout = None):
        if stdout==None:
            self.terminal = sys.stdout
        else:
            self.terminal = stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        os.fsync(self.log)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

class CustomImgDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index]

    def __len__(self):
        return len(self.orig)

inception_model = None
def _init_inception():
    # Load inception model
    global inception_model
    #model_name = 'inceptionresnetv2'
    #inception_model = pretrainedmodels.__dict__[model_name](num_classes=1001, pretrained='imagenet+background')
    #inception_model = torch.nn.DataParallel(inception_model).cuda()
    inception_model = inception_v3(pretrained=True, transform_input=False)
    if torch.cuda.is_available():
        inception_model = torch.nn.DataParallel(inception_model).cuda()
    inception_model.eval()

def inception_score(imgs, batch_size=100, resize=True, cuda=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    if inception_model is None:
        _init_inception()

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        print('.', end='',flush=True)
        batch = batch.type(dtype)
        batchv = Variable(batch,requires_grad=True)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    print(' ')
    # Now compute the mean kl-div
    split_scores = []
    print('Split: ', end='')
    for k in range(splits):
        print(k, end='',flush=True)
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    print(' ')

    return np.mean(split_scores), np.std(split_scores)
