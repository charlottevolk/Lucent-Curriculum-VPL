import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MNASNet(nn.Module):
    """
    MNASNet (mnasnet1_0) with skip connections from each block to the readout.
    Dual-stream architecture for comparing two images.
    """
    def __init__(self):
        super(MNASNet, self).__init__()
        self.backbone = torchvision.models.mnasnet1_0(weights=None)
        self.backbone.classifier = nn.Identity()
        # MNASNet layers: stem, blocks, head (match MNASNet)
        self.stem = self.backbone.layers[0:9]  # up to and including first BN (index 8)
        # Blocks: 6 blocks, each is nn.Sequential
        self.blocks = self.backbone.layers[9:-3]  # blocks are 9 to -4 (exclusive)
        self.head = self.backbone.layers[-3:]  # last 3 layers: Conv2d, BN, ReLU

        self.fc = None
        # Run a dummy forward pass to initialize self.fc
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
            self.fc = nn.Linear(x_flat.shape[1], 1)
            self.fc.weight.data.fill_(0)
            self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x):
        skips = []
        # Stem
        x = self.stem(x)
        s = x.view(x.size(0), -1)
        skips.append(s)
        # Blocks
        for block in self.blocks:
            x = block(x)
            s = x.view(x.size(0), -1)
            skips.append(s)
        # Head
        x = self.head(x)
        s = x.view(x.size(0), -1)
        skips.append(s)
        x_flat = torch.cat(skips, dim=1)
        return x_flat

    def forward(self, x1, x2, sep):
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        x1 = self.extract_skip_features(x1)
        x2 = self.extract_skip_features(x2)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        delta_h = x1 - x2
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)
        return p, delta_h