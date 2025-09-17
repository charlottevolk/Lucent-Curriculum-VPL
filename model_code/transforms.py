# A torchvision transform that adds gaussian noise to an image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import math
#import cv2
import PIL
from PIL import Image
import scipy
from scipy import ndimage


class GaussianNoise(object):
    """
    Add Gaussian noise.
    
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        c, h, w = tensor.size()
        noise = tensor + torch.randn(1, h, w) * self.std + self.mean # if this doesn't work, generate 1 noise pattern & add to each channel separately
        return noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
