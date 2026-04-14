import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops.stochastic_depth import StochasticDepth

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(self.dims)

class CNBlock(nn.Module):
    def __init__(self, dim, layer_scale: float = 1e-6, stochastic_depth_prob: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            Permute((0, 3, 1, 2)),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
    def forward(self, x):
        result = self.layer_scale * self.block(x)
        result = self.stochastic_depth(result)
        result += x
        return result

class ConvNeXt(nn.Module):
    """
    ConvNeXt with skip connections from each stage to the readout.
    Dual-stream architecture for comparing two images.
    """
    def __init__(self, in_chans=3, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 stochastic_depth_prob=0.0, layer_scale=1e-6):
        super().__init__()
        # Build features as nn.Sequential (stem + stages + downsamples), matching ConvNeXt
        features = []
        # Stem
        features.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        ))
        # Stages and downsamples
        total_stages = len(depths)
        for stage_idx in range(total_stages):
            stage = nn.Sequential(*[CNBlock(dims[stage_idx], layer_scale, 0.0) for _ in range(depths[stage_idx])])
            features.append(stage)
            if stage_idx < total_stages - 1:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[stage_idx], eps=1e-6),
                    nn.Conv2d(dims[stage_idx], dims[stage_idx + 1], kernel_size=2, stride=2),
                )
                features.append(downsample)
        self.features = nn.Sequential(*features)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 227, 227)
            x_flat = self.extract_skip_features(dummy)
        self.fc = nn.Linear(x_flat.shape[1], 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def extract_skip_features(self, x):
        skips = []
        for layer in self.features:
            x = layer(x)
            s = x.reshape(x.size(0), -1)
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