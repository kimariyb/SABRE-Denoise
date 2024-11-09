import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import shape_inference

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.main(x)
    

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):   
        return self.main(x)


class SabreNet(nn.Module):
    # 输入数据：[batch, 2, 8192] 其中 2 为通道数
    def __init__(self, in_channels=2, out_channels=2):
        super(SabreNet, self).__init__()
        self.downs = nn.Sequential(
            DownConv(in_channels=in_channels, out_channels=64),
            DownConv(in_channels=64, out_channels=128),
            DownConv(in_channels=128, out_channels=256),
        )        
        
        self.middle = nn.Sequential(    
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.5)
        )
        
        self.ups = nn.Sequential(
            UpConv(in_channels=768, out_channels=256),
            UpConv(in_channels=384, out_channels=128),
            UpConv(in_channels=192, out_channels=64)
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


if __name__ == '__main__':  
    model = SabreNet()
    input = torch.randn(1, 2, 8192)
    output = model(input)
    
    torch.onnx.export(model, input, "SabreNet.onnx", export_params=True, verbose=True)
    
    onnx_model = onnx.load("SabreNet.onnx")
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, "para_SabreNet.onnx")
