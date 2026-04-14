# a pytorch dataset class for loading static grating stimuli from a folder
# and returning them as a torch tensor



import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class GratingDataset(Dataset):
    """Grating dataset."""

    def __init__(self, root_dir, transform=None, img_type='Target', num_images=1000, scaled=False, force_type='none', factor=1.0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """

        self.root_dir = root_dir
        self.transform = transform
        self.type = img_type
        self.num_images = num_images
        self.image_list = os.listdir(root_dir)
        self.CW_list = []
        self.CCW_list = []
        self.scaled = scaled
        self.force_type = force_type
        self.factor = factor

        for filename in self.image_list:
            f = os.path.join(root_dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                # get params from filenames
                params = filename.split("_")
            if 'CCW' in filename:
                self.CCW_list.append(filename)
            elif 'CW' in filename:
                self.CW_list.append(filename)

    def __len__(self):
        return self.num_images

    def set_scaled(self, scaled):
        self.scaled = scaled
    
    def set_force_type(self, force_type):
        self.force_type = force_type

    def set_factor(self, factor):
        self.factor = factor

    def __getitem__(self, idx):
        if self.scaled and self.force_type == 'seq':
            self.image_list = natsorted(self.image_list)
            weights = [1.0, 1.0, self.factor, self.factor]
            image_fn = random.choices(self.image_list, weights)[0]
        elif self.scaled and self.force_type == 'antiseq':
            self.image_list = natsorted(self.image_list)
            weights = [self.factor, self.factor, 1.0, 1.0]
            image_fn = random.choices(self.image_list, weights)[0]
        else:
            CCW_fn = random.choice(self.CCW_list)
            CW_fn = random.choice(self.CW_list)

        CCW_name = os.path.join(self.root_dir, CCW_fn)
        CW_name = os.path.join(self.root_dir, CW_fn)

        # set the label of the image. If the image is rotated CW, the label is 1. If the image is rotated CCW, the label is 0.
        CCW_label = 0
        CW_label = 1

        CCW_sep = CCW_fn.split('_')[4]
        CW_sep = CW_fn.split('_')[4]

        image_CCW = Image.open(CCW_name)
        image_CW = Image.open(CW_name)
        if self.transform:
            image_CCW = self.transform(image_CCW)
            image_CW = self.transform(image_CW)

        images = torch.stack((image_CCW, image_CW))
        labels = torch.tensor([CCW_label, CW_label])
        seps = [CCW_sep, CW_sep]

        
        return images, labels, seps # dim 2 x 3 x w x h (batch dimension is first)
    # if batch size of 10, 2 images in each (CW & CCW), training code will flatten and split into 20 images 10 x 2 x 3 x w x h --> 20 x 3 x w x h


