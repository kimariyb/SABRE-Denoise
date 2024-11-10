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
        
        
def convLx1(cin, cout, L, stride=1, groups=1, bias=False):
    return StdConv1d(cin, cout, kernel_size=L, stride=stride, padding=(L-1)//2, bias=bias, groups=groups)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

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
            
        self.reset_weights()
            
    def reset_weights(self):
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
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv1d(2, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        
        b, c, in_size = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)(x)
        
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2]] = x[:]
            else:
                feat = x
                
            features.append(feat)
            
        x = self.body[-1](x)
        
        return x, features[::-1]
