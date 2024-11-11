import torch
import torch.nn as nn

from torch.nn import functional as F

import numpy as np


class UpsamplingBilinear1d(nn.Module):
    def __init__(self, scale_factor):
        super(UpsamplingBilinear1d, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=True)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )
        
        self.up = UpsamplingBilinear1d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
    

class SpectralDeNoiseHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = UpsamplingBilinear1d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv1d, upsampling)

# TODO: 将 DecoderCup 处理为一维卷积
class DecoderCup(nn.Module):
    def __init__(
        self,
        embedding_dim,
        decoder_channels,
        n_skip,
        skip_channels,
    ):
        super().__init__()
        self.head_channels = 512
        self.conv_more = nn.Sequential(
            nn.Conv1d(embedding_dim, self.head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.head_channels),
        )
        
        self.decoder_channels = decoder_channels
        self.in_channels = [self.head_channels] + list(self.decoder_channels[:-1])
        self.out_channels = self.decoder_channels
        
        self.n_skip = n_skip
        self.skip_channels = skip_channels

        if self.n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
        ]
        
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        B, n_patch, hidden = hidden_states.size()  
        w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            
        return x