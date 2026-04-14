import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        # VGG16 architecture: 13 conv layers + skip connections from every layer to readout
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # For 227x227 input, flattened skip feature total is 15,323,072
        self.fc1 = nn.Linear(15323072, 1)
        self.fc1.weight.data.fill_(0)
        self.fc1.bias.data.fill_(0)

    def forward(self, x1, sep):
        # pass the input through the network and collect skip connections
        skip_x1 = []

        x1 = F.relu(self.conv1_1(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv1_2(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.pool1(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv2_1(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv2_2(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.pool2(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv3_1(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv3_2(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv3_3(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.pool3(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv4_1(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv4_2(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv4_3(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.pool4(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv5_1(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv5_2(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = F.relu(self.conv5_3(x1))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.pool5(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = torch.cat(skip_x1, dim=1)
        del skip_x1, x1_dim
        x1 = self.fc1(x1)

        delta_h = x1
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)

        return p, delta_h