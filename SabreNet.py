import torch
import torch.nn as nn


class SabreNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):