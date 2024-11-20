
import torch
import torch.nn as nn
import torch.nn.functional as F
        

# 用于实现预激活瓶颈结构的类
class PreActBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        stride=1,
    ):
        super().__init__()
        
        # 1x1 卷积层
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        
        self.gn1 = nn.GroupNorm(4, mid_channels, eps=1e-6)

        # 3x3 卷积层
        self.conv2 = nn.Conv1d(
            in_channels=mid_channels, 
            out_channels=mid_channels,
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )  

        self.gn2 = nn.GroupNorm(4, mid_channels, eps=1e-6)
        
        # 1x1 卷积层
        self.conv3 = nn.Conv1d(
            in_channels=mid_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.gn3 = nn.GroupNorm(4, out_channels, eps=1e-6)
        
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        
        if (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.GroupNorm(out_channels, out_channels, eps=1e-6)
            )
            
        self.init_weights()
            
    def init_weights(self):
        # 初始化卷积层的权重和偏置
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='leaky_relu')
        if hasattr(self, 'downsample'):
            nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        
        return y


class ResNet(nn.Module):
    def __init__(self, length=4):
        super().__init__()

        self.length = length

        self.root = nn.Sequential(
            nn.Conv1d(2, self.length, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(4, self.length, eps=1e-6),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            PreActBottleneck(self.length, self.length * 4, self.length),
            PreActBottleneck(self.length * 4, self.length * 4, self.length),
        )

        self.block2 = nn.Sequential(
            PreActBottleneck(self.length * 4, self.length * 8, self.length * 2, stride=2),
            PreActBottleneck(self.length * 8, self.length * 8, self.length * 2),
        )

        self.block3 = nn.Sequential(
            PreActBottleneck(self.length * 8, self.length * 16, self.length * 4, stride=2),
            PreActBottleneck(self.length * 16, self.length * 16, self.length * 4)
        )
        
        self.body = nn.ModuleList([self.block1, self.block2, self.block3])

    def forward(self, x):
        features = []
        
        b, c, l = x.size()         
        x = self.root(x)
        x = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)(x)
                
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



            
