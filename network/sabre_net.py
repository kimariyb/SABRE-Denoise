
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.sabre_layer import DownsamplingBlock, UpsamplingBlock


# TODO: 实现 SabreNet
class SabreNet(nn.Module):
    # 输入数据：[batch, 2, 8192] 其中 2 为通道数
    def __init__(self, in_channels=2, out_channels=2):
        super(SabreNet, self).__init__()
        self.downs = nn.Sequential(
            DownsamplingBlock(in_channels=in_channels, out_channels=64),
            DownsamplingBlock(in_channels=64, out_channels=128),
            DownsamplingBlock(in_channels=128, out_channels=256),
        )        
        
        self.middle = nn.Sequential(    
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.5)
        )
        
        self.ups = nn.Sequential(
            UpsamplingBlock(in_channels=768, out_channels=256),
            UpsamplingBlock(in_channels=384, out_channels=128),
            UpsamplingBlock(in_channels=192, out_channels=64)
        )

        self.out = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        x = self.middle(x)
                    
        for up, skip in zip(self.ups, reversed(skips)):
            x = torch.cat((x, skip), dim=1)
            x = up(x)

        x = self.out(x)
        
        return x