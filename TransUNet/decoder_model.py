import torch
import torch.nn as nn

from torch.nn import functional as F


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
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(out_channels),
        )
        
        self.up = UpsamplingBilinear1d(scale_factor=2)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.normal_(m.bias, std=1e-6)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, skip=None):
        x = self.up(x)
    
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
    

class DecoderCup(nn.Module):
    def __init__(
        self,
        embedding_dim,
        decoder_channels,
        skip_channels,
        skip_num,
    ):
        super().__init__()
        self.head_channels = 512
        
        self.conv_more = nn.Sequential(
            nn.Conv1d(embedding_dim, self.head_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(self.head_channels),
        )
        
        self.decoder_channels = decoder_channels
        self.in_channels = [self.head_channels] + list(self.decoder_channels[:-1])
        self.out_channels = self.decoder_channels
        
        self.skip_channels = skip_channels
        self.skip_num = skip_num

        if self.skip_num != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.skip_num):
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
        ]
        
        self.blocks = nn.ModuleList(blocks)
        
        
    def init_weights(self):
        for block in self.blocks:
            block.reset_parameters()
    
    def forward(self, x, features=None):
        # reshape from (B, n_patch, hidden) to (B, hidden, n_patch)
        B, n_patch, hidden = x.size()  
        x.permute(0, 2, 1)
        x = x.contiguous().view(B, (n_patch * hidden) // 512, 512)  
        x = self.conv_more(x)
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.skip_num) else None
            else:
                skip = None
                
            x = decoder_block(x, skip=skip)

        return x
    

