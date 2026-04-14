import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256*6*6, 1) # for copied weights
        self.fc1.weight.data.fill_(0) # changing all readout weights to zero
        self.fc1.bias.data.fill_(0) # ditto for biases
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x1, x2, sep):
        # pass the two inputs through the network and calculate the difference

        # x1
        x1 = F.relu(self.conv1(x1)) 
        x1 = self.pool(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool(x1)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        x1 = self.pool(x1) 
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1) 
       
        # x2
        x2 = F.relu(self.conv1(x2))
        x2 = self.pool(x2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.pool(x2)
        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        x2 = F.relu(self.conv5(x2))
        x2 = self.pool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)

        # calculate the difference
        delta_h = x1 - x2

        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)

        return p, delta_h



