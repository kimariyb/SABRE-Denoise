from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F



class StdConv1d(nn.Conv1d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv1d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        
        
def convLx1(in_channels, out_channels, length, stride=1, groups=1, bias=False):
    return StdConv1d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=length, 
        stride=stride, 
        padding=(length-1)//2, 
        bias=bias, 
        groups=groups
    )


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout 
        cmid = cmid

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = convLx1(cin, cmid, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = convLx1(cmid, cmid, 3, stride, bias=False)  
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = convLx1(cmid, cout, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = convLx1(cin, cout, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)
            
        self.init_weights()
            
    def init_weights(self):
        # 初始化卷积层的权重和偏置
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(self, 'downsample'):
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')

        # 初始化组归一化层的权重和偏置
        nn.init.ones_(self.gn1.weight)
        nn.init.zeros_(self.gn1.bias)
        nn.init.ones_(self.gn2.weight)
        nn.init.zeros_(self.gn2.bias)
        nn.init.ones_(self.gn3.weight)
        nn.init.zeros_(self.gn3.bias)
        if hasattr(self, 'downsample'):
            nn.init.ones_(self.gn_proj.weight)
            nn.init.zeros_(self.gn_proj.bias)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        
        return y


class ResNet(nn.Module):
    def __init__(self, block_units, length):
        super().__init__()
        
        self.length = length

        self.root = nn.Sequential(
            StdConv1d(2, self.length, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(32, self.length, eps=1e-6),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=self.length, cout=self.length*4, cmid=self.length))] +
                [(f'unit{i:d}', PreActBottleneck(cin=self.length*4, cout=self.length*4, cmid=self.length)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=self.length*4, cout=self.length*8, cmid=self.length*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=self.length*8, cout=self.length*8, cmid=self.length*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=self.length*8, cout=self.length*16, cmid=self.length*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=self.length*16, cout=self.length*16, cmid=self.length*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

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
        self.model = ResNet(block_units, width_factor)

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

    