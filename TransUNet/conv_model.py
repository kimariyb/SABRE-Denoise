
import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于实现标准卷积层的类
class StdConv1d(nn.Conv1d):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros', 
        device=None,
        dtype=None
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x):
        # 标准卷积层的实现
        # 先对输入进行标准化，即减去均值，除以标准差
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        
        return F.conv1d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        

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
        self.conv1 = StdConv1d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        
        self.gn1 = nn.GroupNorm(4, mid_channels, eps=1e-6)

        # 3x3 卷积层
        self.conv2 = StdConv1d(
            in_channels=mid_channels, 
            out_channels=mid_channels,
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )  

        self.gn2 = nn.GroupNorm(4, mid_channels, eps=1e-6)
        
        # 1x1 卷积层
        self.conv3 = StdConv1d(
            in_channels=mid_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.gn3 = nn.GroupNorm(4, out_channels, eps=1e-6)
        
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        if (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                StdConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.GroupNorm(out_channels, out_channels, eps=1e-6)
            )
            
        self.init_weights()
            
    def init_weights(self):
        # 初始化卷积层的权重和偏置
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(self, 'downsample'):
            nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_in', nonlinearity='leaky_relu')

        nn.init.normal_(self.conv1.bias, std=1e-6)
        nn.init.normal_(self.conv2.bias, std=1e-6)
        nn.init.normal_(self.conv3.bias, std=1e-6)
        if hasattr(self, 'downsample'):
            nn.init.normal_(self.downsample[1].bias, std=1e-6)

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
            StdConv1d(2, self.length, kernel_size=7, stride=2, padding=3, bias=False),
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



            
