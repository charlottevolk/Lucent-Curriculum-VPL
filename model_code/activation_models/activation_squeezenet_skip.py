import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x1):
        x1 = self.squeeze_activation(self.squeeze(x1))
        return self.expand_activation(torch.cat([
            self.expand1x1(x1),
            self.expand3x3(x1)
        ], 1))


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        self.skip_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        self.dropout = nn.Dropout(p=0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x1):
        skip_x1 = []
        out = x1

        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in self.skip_indices:
                skip_x1.append(out.view(out.size(0), -1))

        x1 = torch.cat(skip_x1, dim=1)
        return x1

    def forward(self, x1, sep):
        x1 = self.extract_skip_features(x1)
        x1 = self.dropout(x1)
        x1 = self.fc(x1)

        delta_h = x1
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)

        return p, delta_h
