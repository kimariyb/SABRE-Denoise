
import netron
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import shape_inference

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, padding=7, stride=1):
        super(DownConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01) 
        )

    def forward(self, x):
        return self.main(x)
    

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=1, stride=2):
        super(UpConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True) 
        )
    
    def forward(self, x):   
        return self.main(x)


class SabreNet(nn.Module):
    # 输入数据：[batch, 2, 8192] 其中 2 为通道数
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]):
        super(SabreNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for feature in features:
            self.downs.append(DownConv(in_channels, feature))
            in_channels = feature

        for feature in features[::-1]:
            self.ups.append(UpConv(in_channels, feature))
            self.ups.append(nn.Conv1d(feature*2, feature, kernel_size=1))
            in_channels = feature
            
        self.bottleneck = nn.Sequential(
            nn.Conv1d(features[-1], features[-1], kernel_size=15, padding=7, stride=1),
            nn.BatchNorm1d(features[-1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True) 
        )
        
        self.out = nn.Sequential(
            nn.Conv1d(features[0], out_channels, kernel_size=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)            
            skip_connections.append(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]   
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return self.out(x)


        
if __name__ == '__main__':  
    model = SabreNet()
    input = torch.randn(1, 2, 8192)
    output = model(input)
    
    torch.onnx.export(model, input, "SabreNet.onnx", export_params=True, verbose=True)
    
    onnx_model = onnx.load("SabreNet.onnx")
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, "para_SabreNet.onnx")
