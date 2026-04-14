# a pytorch dataset class for loading static grating stimuli from a folder
# and returning them as a torch tensor



import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from natsort import natsorted

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
        # This is only relevant if using forced_seq or forced_antiseq flags
        if self.scaled and self.force_type == 'seq':
            self.image_list = natsorted(self.image_list)
            weights = [1.0, 1.0, self.factor, self.factor]
            image_fn = random.choices(self.image_list, weights)[0]
        elif self.scaled and self.force_type == 'antiseq':
            self.image_list = natsorted(self.image_list)
            weights = [self.factor, self.factor, 1.0, 1.0]
            image_fn = random.choices(self.image_list, weights)[0]
        else:
            # most of the time this will be used - random choice of images
            image_fn = random.choice(self.image_list)
        image_name = os.path.join(self.root_dir, image_fn)

        # set the label of the image. If the image is rotated CW, the label is 1. If the image is rotated CCW, the label is 0.
        if 'CCW' in image_name:
            image_label = 0
        elif 'CW' in image_name:
            image_label = 1

        image_sep = image_fn.split('_')[4]

        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(image_label), image_sep # image dim 1 x 3 x w x h (batch dimension is first)

