
import netron
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.onnx import export


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, padding=7, stride=1):
        super(DownSample, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.main(x)
    

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=1):
        super(UpSample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)  
        )

    def forward(self, x):
        return self.main(x)


class SabreNet(nn.Module):
    # 输入数据：[batch, 1, 8192, 2] y: 1 x: 8192 z: 2，其中 2 为通道数
    def __init__(self):
        super(SabreNet, self).__init__()
        self.encoder = nn.Sequential(
            DownSample(in_channels=2, out_channels=32, kernel_size=3, padding=1, stride=1),
            DownSample(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            DownSample(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            DownSample(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            DownSample(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),           
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    
        self.decoder = nn.Sequential(
            UpSample(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
            UpSample(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            UpSample(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            UpSample(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.output(x)
        
        return x
        
        
if __name__ == '__main__': 
    model = SabreNet()
    input = torch.randn(1, 2, 8192, 1)
    output = model(input)

    export(model, input, "SabreNet.onnx", verbose=True)
    netron.start("SabreNet.onnx")
    