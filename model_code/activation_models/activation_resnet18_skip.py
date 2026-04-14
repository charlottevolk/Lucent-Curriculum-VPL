import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # For 227x227 input: concat(conv1, maxpool, layer1, layer2, layer3, layer4) = 1,445,632
        self.fc = nn.Linear(1445632, 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x1, sep):
        skip_x1 = []

        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.maxpool(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.layer1(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.layer2(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.layer3(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = self.layer4(x1)
        x1_dim = x1.shape
        skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3]))

        x1 = torch.cat(skip_x1, dim=1)
        del skip_x1, x1_dim
        x1 = self.fc(x1)

        delta_h = x1
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)

        return p, delta_h