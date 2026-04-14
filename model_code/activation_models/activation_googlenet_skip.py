import torch
import torch.nn as nn
import torchvision


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.backbone = torchvision.models.googlenet(weights=None, aux_logits=False)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x1):
        skip_x1 = []

        x1 = self.backbone.conv1(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.maxpool1(x1)
        x1 = self.backbone.conv2(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.conv3(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.maxpool2(x1)

        x1 = self.backbone.inception3a(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.inception3b(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.maxpool3(x1)

        x1 = self.backbone.inception4a(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.inception4b(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.inception4c(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.inception4d(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.inception4e(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.maxpool4(x1)

        x1 = self.backbone.inception5a(x1)
        skip_x1.append(x1.view(x1.size(0), -1))

        x1 = self.backbone.inception5b(x1)
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
