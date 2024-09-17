import torch
import torch.nn as nn


def complex_init(fan_in, fan_out, seed, criterion='glorot'):
    rng = torch.Generator()
    rng.manual_seed(seed)
    if criterion == 'glorot':
        s = 1. / torch.sqrt(fan_in + fan_out)
    elif criterion == 'he':
        s = 1. / torch.sqrt(fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    modulus = torch.rayleigh(torch.empty(fan_out, fan_in), scale=s, generator=rng)
    phase = torch.empty(fan_out, fan_in).uniform_(-torch.pi, torch.pi)
    
    return modulus * torch.cos(phase), modulus * torch.sin(phase)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, 
                 activation=True, criterion='he', seed=1337
    ):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.activation = activation
        self.criterion = criterion
        self.seed = seed
        
        self.weight_real = nn.Parameter(torch.Tensor(out_channels, in_channels // 2, *kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(out_channels, in_channels // 2, *kernel_size))