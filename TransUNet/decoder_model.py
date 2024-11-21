import torch
import torch.nn as nn

from torch.nn import functional as F


class UpsamplingBilinear1d(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
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
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(out_channels),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(out_channels),
        )
        
        self.up = UpsamplingBilinear1d(scale_factor=2)  # Using linear interpolation for 1D data
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1) 

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)  
        
        return x
    

class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.root = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),
        )
        
        self.body = nn.ModuleList([
            DecoderBlock(in_channels=256, out_channels=128, skip_channels=128),
            DecoderBlock(in_channels=128, out_channels=64, skip_channels=64),
            DecoderBlock(in_channels=64, out_channels=32, skip_channels=32),
            DecoderBlock(in_channels=32, out_channels=16, skip_channels=16),
        ])
        
        self.denoiser = DecoderBlock(in_channels=16, out_channels=1, skip_channels=0) 
    
    def forward(self, x, features=None):
        # reshape from (B, n_patch, hidden) to (B, hidden, n_patch)
        B, n_patch, hidden = x.size()  
        
        x.permute(0, 2, 1)
        x = x.contiguous().view(B, (n_patch * hidden) // 256, 256)  
        
        x = self.root(x)  

        for i, block in enumerate(self.body):
            skip = features[i] if features is not None else None
            x = block(x, skip=skip) 
            
        x = self.denoiser(x)  # Final denoising layer
            
        return x
    

