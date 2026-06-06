"""
1D DnCNN baseline adapted from Zhang et al. 2017.
"""
import torch
import torch.nn as nn


class DnCNN1D(nn.Module):
    """
    1D DnCNN for complex NMR spectrum denoising.

    Input shape : (B, 2, N)
                   channel 0 = real part of FID/spectrum
                   channel 1 = imaginary part of FID/spectrum
    Output shape: same as input (residual learning: output = input - noise)

    Architecture (Zhang et al. 2017):
        Layer 1        : Conv(bias=True) + ReLU
        Layers 2 ~ D-1 : Conv(bias=False) + BN + ReLU
        Layer D        : Conv(bias=True)
    """

    def __init__(self, in_channels: int = 2, depth: int = 17, n_feats: int = 64):
        """
        Args:
            in_channels: 输入/输出通道数。复数信号拆成实/虚两通道，故默认为 2。
            depth:       网络总层数，须 >= 2。原论文高斯去噪用 17，盲去噪用 20。
            n_feats:     中间层通道数。原论文为 64；若显存充裕可适当增大。
        """
        super().__init__()

        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")

        layers = [
            nn.Conv1d(in_channels, n_feats, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.extend([
                nn.Conv1d(n_feats, n_feats, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(n_feats),
                nn.ReLU(inplace=True),
            ])

        layers.append(nn.Conv1d(n_feats, in_channels, kernel_size=3, padding=1, bias=True))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy spectrum, shape (B, 2, N)
               dim-1: [real, imag]
        Returns:
            denoised spectrum, same shape
            residual learning: denoised = noisy - predicted_noise
        """
        return x - self.net(x)


if __name__ == '__main__':
    net = DnCNN1D(in_channels=2, depth=20, n_feats=256)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"DnCNN1D params: {n_params / 1e6:.3f} M")

    x = torch.randn(4, 2, 8192)
    y = net(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    assert x.shape == y.shape, "Output shape mismatch!"
    print("Shape check passed.")

    try:
        _ = DnCNN1D(depth=1)
    except ValueError as e:
        print(f"Caught expected error: {e}")