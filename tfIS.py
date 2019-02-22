from is_utils import get_inception_score
import numpy as np

fask_imgs = np.load("fakeimgs_final.npy")
print ("Calculating Inception Score for fakeimgs_final...")
print(get_inception_score(fask_imgs))
fask_imgs = np.load("fakeimgs_best.npy")
print ("Calculating Inception Score for fakeimgs_best...")
print(get_inception_score(fask_imgs))
