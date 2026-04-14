import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

class SqueezeNet(nn.Module):
    """
    SqueezeNet with skip connections from each block to the readout.
    Dual-stream architecture for comparing two images.
    Structure matches torchvision SqueezeNet for compatibility.
    """
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),      # in: 64, out: 128
            Fire(128, 16, 64, 64),     # in: 128, out: 128
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),   # in: 128, out: 256
            Fire(256, 32, 128, 128),   # in: 256, out: 256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),   # in: 256, out: 384
            Fire(384, 48, 192, 192),   # in: 384, out: 384
            Fire(384, 64, 256, 256),   # in: 384, out: 512
            Fire(512, 64, 256, 256),   # in: 512, out: 512
        )
        # Indices of layers after which to take skip connections (after each block)
        self.skip_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.dropout = nn.Dropout(p=0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x):
        skips = []
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in self.skip_indices:
                s = out.view(out.size(0), -1)
                skips.append(s)
        x_flat = torch.cat(skips, dim=1)
        return x_flat

    def forward(self, x1, x2, sep):
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        x1 = self.extract_skip_features(x1)
        x1 = self.dropout(x1)
        x1 = self.fc(x1)
        x2 = self.extract_skip_features(x2)
        x2 = self.dropout(x2)
        x2 = self.fc(x2)
        delta_h = x1 - x2
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)
        return p, delta_h