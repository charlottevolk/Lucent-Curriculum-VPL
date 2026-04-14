import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class EfficientNet(nn.Module):
    """
    EfficientNet (B0) with skip connections from each block to the readout.
    Dual-stream architecture for comparing two images.
    """
    def __init__(self):
        super(EfficientNet, self).__init__()
        # Load EfficientNet backbone
        self.backbone = torchvision.models.efficientnet_b0(weights=None)
        # Remove classifier, keep features
        self.backbone.classifier = nn.Identity()
        # Get the blocks (stages) of EfficientNet
        self.stem = self.backbone.features[0]  # Conv2d stem
        self.blocks = self.backbone.features[1:-1]  # MBConv blocks (6 blocks for B0)
        self.head = self.backbone.features[-1]  # Final Conv2d
        # The fc layer will be initialized after a dummy forward pass with the correct input size
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
        # Concatenate all skip features
        x_flat = torch.cat(skips, dim=1)
        return x_flat

    def forward(self, x1, x2, sep):
        # Add batch dimension to x2 (reference image) if needed
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        x1 = self.extract_skip_features(x1)
        x2 = self.extract_skip_features(x2)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        # calculate the difference
        delta_h = x1 - x2
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)
        return p, delta_h