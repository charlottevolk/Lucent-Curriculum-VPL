import torch
import torch.nn as nn
import torchvision


class MNASNet(nn.Module):
    def __init__(self):
        super(MNASNet, self).__init__()
        self.backbone = torchvision.models.mnasnet1_0(weights=None)
        self.backbone.classifier = nn.Identity()

        self.stem = self.backbone.layers[0:9]
        self.blocks = self.backbone.layers[9:-3]
        self.head = self.backbone.layers[-3:]

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x1):
        skip_x1 = []

        x1 = self.stem(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        for block in self.blocks:
            x1 = block(x1)
            skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.head(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = torch.cat(skip_x1, dim=1)
        return x1

    def forward(self, x1, sep):
        x1 = self.extract_skip_features(x1)
        x1 = self.fc(x1)

        delta_h = x1
        # pass the difference through a sigmoid
        p = torch.sigmoid(delta_h)
        return p, delta_h
