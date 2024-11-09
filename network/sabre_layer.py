import torch
import torch.nn as nn


# TODO: 后期继续改进
class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownsamplingBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.dropout = nn.Dropout(p=0.5)
        
        self.reset_param()

    def reset_param(self):
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
        
        
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpsamplingBlock, self).__init__()

        self.trans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.reset_param()

    def reset_param(self):
        nn.init.kaiming_normal_(self.trans.weight)
        nn.init.constant_(self.trans.bias, 0)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.trans(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x