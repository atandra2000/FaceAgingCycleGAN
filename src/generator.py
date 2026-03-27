"""CycleGAN Generator for Face Aging
Industry-standard ResNet with attention, adaptive conditioning, and spectral norm."""

import torch.nn as nn
import torch.nn.functional as F
from modules import SelfAttention


class AdaptiveResidualBlock(nn.Module):
    """Residual block with AdaIN for conditioning and spectral norm.
    """
    def __init__(self, in_features, dropout_rate=0.0):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_features, in_features, 3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_features, in_features, 3, padding=1))
        self.norm1 = nn.InstanceNorm2d(in_features)
        self.norm2 = nn.InstanceNorm2d(in_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def adain(self, x, style):
        """Adaptive Instance Normalization"""
        if style is None:
            return x
        
        # Normalize input
        mean = x.mean([2, 3], keepdim=True)
        std = x.std([2, 3], keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        
        # Extract style mean and std
        if style.dim() == 4:
            # Style is [B, C, H, W]
            style_mean = style.mean((2, 3), keepdim=True)
            style_std = style.std((2, 3), keepdim=True) + 1e-8
        else:
            # Style is [B, 2*C] - split into mean and std
            style_mean, style_std = style.chunk(2, 1)
            style_mean = style_mean.view(style_mean.size(0), -1, 1, 1)
            style_std = style_std.view(style_std.size(0), -1, 1, 1)
        
        return style_std * x_norm + style_mean

    def forward(self, x, style=None):
        residual = x
        
        # Conv → InstanceNorm → AdaIN → ReLU → Conv → AdaIN → Dropout
        residual = self.conv1(residual)
        residual = self.norm1(residual)
        if style is not None:
            residual = self.adain(residual, style)
        residual = F.relu(residual, inplace=True)
        
        residual = self.conv2(residual)
        residual = self.norm2(residual)
        if style is not None:
            residual = self.adain(residual, style)
        residual = self.dropout(residual)
        
        return x + residual


class ConditionalGenerator(nn.Module):
    """Conditional generator with attention and AdaIN.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_residual_blocks=10, 
                 num_ages=101, dropout_rate=0.1):
        super(ConditionalGenerator, self).__init__()
        self.num_ages = num_ages
        self.ngf = ngf
        self.n_residual_blocks = n_residual_blocks
        
        # Age embedding and style MLP
        self.age_embedding = nn.Embedding(num_ages, ngf * 8)
        self.style_mlps = nn.ModuleList([
            nn.Sequential(
            nn.Linear(ngf * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * ngf * 4))for _ in range(n_residual_blocks)])

        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, 7)),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.downsample = nn.ModuleList()
        in_features = ngf
        for _ in range(2):
            out_features = in_features * 2
            self.downsample.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features

        # Residual blocks with AdaIN
        self.residuals = nn.ModuleList([
            AdaptiveResidualBlock(in_features, dropout_rate) 
            for _ in range(n_residual_blocks)
        ])

        # Self-attention after residuals
        self.attention = SelfAttention(in_features)

        # Upsampling
        self.upsample = nn.ModuleList()
        for _ in range(2):
            out_features = in_features // 2
            self.upsample.append(nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, 
                    padding=1, output_padding=1
                )),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features

        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, 7)),
            nn.Tanh()
        )

    def forward(self, x, age):
        # Get style codes from age embedding
        age_emb = self.age_embedding(age)  # [B, ngf*8]
        style_per_layer = [mlp(age_emb) for mlp in self.style_mlps]  # List of [B, 2*ngf*8]
        
        out = self.initial(x)

        # Downsampling
        for layer in self.downsample:
            out = layer(out)

        # Residuals with per-layer AdaIN
        for i, res in enumerate(self.residuals):
            out = res(out, style_per_layer[i])  # [B, 2*ngf*8]

        # Attention
        out = self.attention(out)

        # Upsampling
        for layer in self.upsample:
            out = layer(out)

        # Output
        return self.output(out)