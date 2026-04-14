import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MobileNetV3(nn.Module):
    """
    MobileNetV3 (Small) with skip connections from each block to the readout.
    Dual-stream architecture for comparing two images.
    """
    def __init__(self, version='small'):
        super(MobileNetV3, self).__init__()
        if version == 'large':
            self.backbone = torchvision.models.mobilenet_v3_large(weights=None)
        else:
            self.backbone = torchvision.models.mobilenet_v3_small(weights=None)
        self.backbone.classifier = nn.Identity()
        self.blocks = self.backbone.features
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x):
        skips = []
        for block in self.blocks:
            x = block(x)
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