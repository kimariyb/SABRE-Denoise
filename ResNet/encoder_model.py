
import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv1d(nn.Conv1d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2], keepdim=True)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        

# 用于实现预激活瓶颈结构的类
class PreActBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride=1,
    ):
        super().__init__()
        
        # 1x1 卷积层, in_channels -> mid_channels
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(mid_channels)

        # 3x3 卷积层, mid_channels -> mid_channels
        self.conv2 = StdConv1d(
            in_channels=mid_channels, 
            out_channels=mid_channels,
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )  

        self.bn2 =nn.BatchNorm1d(mid_channels)
        
        # 1x1 卷积层, mid_channels -> out_channels
        self.conv3 = StdConv1d(
            in_channels=mid_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.bn3 = nn.BatchNorm1d(out_channels)

        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        if (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                StdConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)

        # Unit's branch
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        
        y = self.act(residual + y)
        
        return y


class ResNet(nn.Module):
    """
    ResNet Model for 1D signals.
    """
    def __init__(self):
        super().__init__()

        self.root = nn.Sequential(
            StdConv1d(in_channels=1, out_channels=8, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(num_features=8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.body = nn.ModuleList([
            PreActBottleneck(in_channels=8, mid_channels=4, out_channels=16),
            PreActBottleneck(in_channels=16, mid_channels=8, out_channels=32, stride=2),
            PreActBottleneck(in_channels=32, mid_channels=16, out_channels=64, stride=2),
            PreActBottleneck(in_channels=64, mid_channels=32, out_channels=128, stride=2),
            PreActBottleneck(in_channels=128, mid_channels=64, out_channels=256, stride=2),
        ])
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0) 
        
    def forward(self, x):
        features = []
        b, c, l = x.size()         
        x = self.root(x)
        x = self.pool(x)

        for i in range(len(self.body)):
            x = self.body[i](x)
            right_size = l // (2 ** (i + 1))
            
            if x.size()[2] != right_size:
                feat = torch.zeros((b, x.size()[1], right_size), device=x.device)
                feat[:, :, 0:x.size()[2]] = x[:]
            else:
                feat = x
            features.append(feat)

        return x, features[::-1]



            
