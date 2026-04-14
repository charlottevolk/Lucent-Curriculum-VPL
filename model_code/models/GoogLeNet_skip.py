import torch
import torch.nn as nn
import torchvision


class GoogLeNet(nn.Module):
    """
    Torchvision GoogLeNet backbone with skip connections from major stages to a shared readout.
    Dual-stream architecture for comparing two images.
    """

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.backbone = torchvision.models.googlenet(weights=None, aux_logits=False)

        # Initialize readout width from full flattened skip tensors (skip_alexnet-style).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x):
        skips = []

        x = self.backbone.conv1(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.maxpool1(x)
        x = self.backbone.conv2(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.conv3(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.maxpool2(x)

        x = self.backbone.inception3a(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.inception3b(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.maxpool3(x)

        x = self.backbone.inception4a(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.inception4b(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.inception4c(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.inception4d(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.inception4e(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.maxpool4(x)

        x = self.backbone.inception5a(x)
        skips.append(x.view(x.size(0), -1))

        x = self.backbone.inception5b(x)
        skips.append(x.view(x.size(0), -1))

        return torch.cat(skips, dim=1)

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
