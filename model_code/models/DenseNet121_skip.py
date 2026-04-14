import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x

class DenseNet121(nn.Module):
    """
    DenseNet-121 with skip connections from each block to the readout.
    Dual-stream architecture for comparing two images.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet121, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, 
                                                     kernel_size=7, stride=2, 
                                                     padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        num_features = num_init_features
        # Each denseblock
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                   num_output_features=num_features // 2)
                self.blocks.append(trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x):
        skips = []
        # Initial conv
        x = self.features.conv0(x)
        s = x.view(x.size(0), -1)
        skips.append(s)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        for block in self.blocks:
            x = block(x)
            s = x.view(x.size(0), -1)
            skips.append(s)
        x = self.features.norm5(x)
        s = x.view(x.size(0), -1)
        skips.append(s)
        x_flat = torch.cat(skips, dim=1)
        return x_flat

    def forward(self, x1, x2, sep):
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        x1 = self.extract_skip_features(x1)
        x1 = self.fc(x1)
        x2 = self.extract_skip_features(x2)
        x2 = self.fc(x2)
        delta_h = x1 - x2
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)
        return p, delta_h