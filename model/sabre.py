import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )       
        
    def forward(self, x):        
        return self.conv(x)
    

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
    ):
        super().__init__()
        self.skip_channels = skip_channels
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels+skip_channels, 
                out_channels=out_channels, kernel_size=3, 
                stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
    def forward(self, x, skip=None):
        if skip is not None:
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=True)
            x = torch.cat([x, skip], dim=1) 
        return self.conv_transpose(x)
    

class SabreNet(nn.Module):
    def __init__(self):
        super(SabreNet, self).__init__()

        self.encoder = nn.ModuleList([
            EncoderBlock(in_channels=1, out_channels=16), # (1,8192) -> (16,4096)
            EncoderBlock(in_channels=16, out_channels=32), # (16,4096) -> (32,2048)
            EncoderBlock(in_channels=32, out_channels=64), # (32,2048) -> (64,1024)
            EncoderBlock(in_channels=64, out_channels=128), # (64,1024) -> (128,512)
            EncoderBlock(in_channels=128, out_channels=256), # (128,512) -> (256,256)
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # (256,256) -> (512,256)
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.decoder = nn.ModuleList([
            DecoderBlock(in_channels=512, out_channels=256, skip_channels=256), # (512,256) -> (256,512)
            DecoderBlock(in_channels=256, out_channels=128, skip_channels=128), # (256,512) -> (128,1024)
            DecoderBlock(in_channels=128, out_channels=64, skip_channels=64), # (128,1024) -> (64,2048)
            DecoderBlock(in_channels=64, out_channels=32, skip_channels=32), # (64,2048) -> (32,4096)
            DecoderBlock(in_channels=32, out_channels=16, skip_channels=16), # (32,4096) -> (16,8192)
        ])
        
        # 3x3 conv to output the final image
        self.output = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1), # (16,8192) -> (1,8192)
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skip_connections = []
        
        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        
        for i, block in enumerate(self.decoder):
            skip = skip_connections[-(i+1)]
            x = block(x, skip)
        
        x = self.output(x)
        
        return x


