"""CycleGAN Discriminator for Face Aging
Industry-standard PatchGAN with attention, spectral norm, and multi-scale age awareness."""

import torch.nn as nn
import torch.nn.functional as F
from modules import SelfAttention


class DiscriminatorBlock(nn.Module):
    """Building block for discriminators with spectral norm and optional attention."""
    def __init__(self, in_channels, out_channels, stride=2, use_attention=False, dropout_rate=0.0):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1))
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.attention = SelfAttention(out_channels) if use_attention else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.attention(x)
        return self.dropout(x)


class MultiscaleAgeAwareDiscriminator(nn.Module):
    """Multi-scale discriminator with age awareness and attention."""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_ages=101, num_scales=3, 
                 dropout_rate=0.1, use_attention=True):
        super(MultiscaleAgeAwareDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.age_embedding = nn.Embedding(num_ages, ndf * 8)  # For adaptive fusion in age head

        # Shared feature extractor backbone per scale
        self.feature_extractors = nn.ModuleList()
        self._scale_channels = []

        for _ in range(num_scales):
            layers = []
            in_channels = input_nc
            for layer in range(n_layers + 1):  # +1 to include final layer
                out_channels = min(ndf * (2 ** layer), ndf * 8)
                stride = 1 if layer == n_layers else 2
                layers.append(DiscriminatorBlock(in_channels, out_channels, stride, 
                                                use_attention and layer > 0, dropout_rate))
                in_channels = out_channels

            # in_channels now equals the final out_channels for this extractor
            self.feature_extractors.append(nn.Sequential(*layers))
            self._scale_channels.append(in_channels)

        # Real/fake heads (per scale)
        self.real_fake_heads = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
            for ch in self._scale_channels
        ])

        # Shared age prediction head
        finest_ch = self._scale_channels[0] if len(self._scale_channels) > 0 else min(ndf * (2 ** n_layers), ndf * 8)
        self.age_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(finest_ch, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.utils.spectral_norm(nn.Linear(512, num_ages))
        )

    def forward(self, x, age=None):
        results = []  # List of real/fake predictions per scale
        features = []  # List of intermediate features per scale
        current_x = x

        # Finest scale
        finest_features = self.feature_extractors[0](x)
        real_fake_pred = self.real_fake_heads[0](finest_features)
        results.append(real_fake_pred)
        features.append(finest_features)

        # Coarser scales
        for i in range(1, self.num_scales):
            # Dynamic downsampling
            target_h = current_x.shape[2] // 2
            target_w = current_x.shape[3] // 2
            current_x = F.adaptive_avg_pool2d(current_x, (target_h, target_w))
            
            scale_features = self.feature_extractors[i](current_x)
            real_fake_pred = self.real_fake_heads[i](scale_features)
            results.append(real_fake_pred)
            features.append(scale_features)

        # Age prediction (from finest scale)
        age_pred = None
        if age is not None:
            age_emb = self.age_embedding(age)  # [B, ndf*8]
            age_emb_spatial = age_emb.view(age.size(0), -1, 1, 1)  # [B, ndf*8, 1, 1]
            
            age_emb_expanded = age_emb_spatial.expand_as(finest_features)  # Broadcast to match
            
            fused_features = finest_features + age_emb_expanded  # Adaptive fusion
            age_pred = self.age_head(fused_features)

        return results, features, age_pred  # preds: list[Tensor], features: list[Tensor], age_pred: Tensor or None