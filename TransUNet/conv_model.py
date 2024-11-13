
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


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
        
        self.gn1 = nn.GroupNorm(32, mid_channels, eps=1e-6)

        # 3x3 卷积层
        self.conv2 = StdConv1d(
            in_channels=mid_channels, 
            out_channels=mid_channels,
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )  

        self.gn2 = nn.GroupNorm(32, mid_channels, eps=1e-6)
        
        # 1x1 卷积层
        self.conv3 = StdConv1d(
            in_channels=mid_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.gn3 = nn.GroupNorm(32, out_channels, eps=1e-6)
        
        self.relu = nn.ReLU(inplace=True)
        
        if (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                StdConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.GroupNorm(out_channels, out_channels, eps=1e-6)
            )
            
        self.init_weights()
            
    def init_weights(self):
        # 初始化卷积层的权重和偏置
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(self, 'downsample'):
            nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_out', nonlinearity='relu')

        # 初始化组归一化层的权重和偏置
        nn.init.ones_(self.gn1.weight)
        nn.init.zeros_(self.gn1.bias)
        nn.init.ones_(self.gn2.weight)
        nn.init.zeros_(self.gn2.bias)
        nn.init.ones_(self.gn3.weight)
        nn.init.zeros_(self.gn3.bias)
        if hasattr(self, 'downsample'):
            nn.init.ones_(self.downsample[1].weight)
            nn.init.zeros_(self.downsample[1].bias)

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
    def __init__(self, length_factor=1, num_layers=3):
        super().__init__()

        self.length = int(64 * length_factor)
        self.block_units = [num_layers, num_layers, num_layers]

        self.root = nn.Sequential(
            StdConv1d(2, self.length, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, self.length, eps=1e-6),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(OrderedDict(
            [('unit1', PreActBottleneck(self.length, self.length * 4, self.length))] + 
            [(f'unit{i:d}', PreActBottleneck(self.length * 4, self.length * 4, self.length)) for i in range(2, self.block_units[0] + 1)]
        ))

        self.block2 = nn.Sequential(OrderedDict(
            [('unit1', PreActBottleneck(self.length * 4, self.length * 8, self.length * 2, stride=2))] + 
            [(f'unit{i:d}', PreActBottleneck(self.length * 8, self.length * 8, self.length * 2)) for i in range(2, self.block_units[1] + 1)]
        ))

        self.block3 = nn.Sequential(OrderedDict(
            [('unit1', PreActBottleneck(self.length * 8, self.length * 16, self.length * 4, stride=2))] + 
            [(f'unit{i:d}', PreActBottleneck(self.length * 16, self.length * 16, self.length * 4)) for i in range(2, self.block_units[2] + 1)]
        ))
        
        self.body = nn.ModuleList([self.block1, self.block2, self.block3])

    def forward(self, x):
        features = []
        
        b, c, in_size = x.size()         
        x = self.root(x)
        x = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)(x)
        
        for i in range(len(self.body) - 1):
            x = self.body[i](x)

            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size), device=x.device)
                feat[:, :, 0:x.size()[2]] = x[:]
            else:
                feat = x
                
            features.append(feat)
            
        x = self.body[-1](x)
        
        return x, features[::-1]


class TestResNet:
    def __init__(self, block_units, width_factor):
        self.model = ResNet(width_factor, block_units)

    def test_forward(self):
        # 创建一个随机输入张量，形状为 (batch_size, channels, length)
        batch_size = 1
        channels = 2
        length = 8192  # 输入序列的长度
        input_tensor = torch.randn(batch_size, channels, length)

        print("输入张量的形状: ", input_tensor.shape)

        # 执行前向传播
        output, features = self.model(input_tensor)
        
        print("输出张量的形状: ", output.shape)
        print("特征张量的形状: ", features[0].shape, features[1].shape)
        
        # 验证输出形状
        assert output.shape[0] == batch_size, "输出batch大小不匹配"
        

# 使用示例
if __name__ == "__main__":
    block_units = [2, 2, 2]  # 每个 block 中的bottleneck单元数量
    width_factor = 1  # 宽度因子
    test_resnet = TestResNet(block_units, width_factor)
    test_resnet.test_forward()

            
