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

    def __init__(self, root_dir, transform=None, img_type='Target', ref_orientation=0, separation_angle=10, contrast=1, phase=0, spatial_freq=10, num_images=1000):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ref_orientation (int): The orientation of the reference grating
            separation_angle (int): The separation angle between the reference and test grating
            contrast (int): The contrast of the grating
            phase (int): The phase of the grating
            spatial_freq (int): The spatial frequency of the grating

        """
        self.num_images = num_images
        self.image_list = os.listdir(root_dir)
        self.transform = transform
        self.root_dir = root_dir


    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, random.choice(self.image_list))
      
        # set the label of the image. If the image is rotated CW, the label is 1. If the image is rotated CCW, the label is 0.
        img1_label = np.random.randint(2)

        image_1 = Image.open(img1_name).convert('RGB')
        if self.transform:
            image_1 = self.transform(image_1)

        images = image_1
        labels = torch.tensor(img1_label)

        return images, torch.tensor(labels)


