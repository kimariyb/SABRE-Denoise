"""
Real-valued 1D U-Net baseline — pure PyTorch, no SabreSDN modules.

Standard U-Net with double-conv blocks, stride-2 down/up-sampling,
and plain concatenation skip connections.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution: Conv1d → BN → ReLU → Conv1d → BN → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    """Strided conv downsampling + double conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Transposed conv upsampling → concat skip → double conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RealUNet(nn.Module):
    """
    Standard 1D U-Net for NMR denoising.

    Encoder: 16→32→64→128→256→256  with stride-2 downsampling.
    Decoder: mirrored upsampling with skip connections.
    """
    def __init__(self, in_channels=2, out_channels=2, base_ch=16):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4,
             base_ch * 8, base_ch * 16, base_ch * 16]

        self.inc = ConvBlock(in_channels, c[0])          # [B, 32, 8192]

        self.down1 = DownBlock(c[0], c[1])               # /2
        self.down2 = DownBlock(c[1], c[2])               # /4
        self.down3 = DownBlock(c[2], c[3])               # /8
        self.down4 = DownBlock(c[3], c[4])               # /16
        self.down5 = DownBlock(c[4], c[5])               # /32

        self.bottleneck = ConvBlock(c[5], c[5])

        self.up1 = UpBlock(c[5], c[4])                   # /16
        self.up2 = UpBlock(c[4], c[3])                   # /8
        self.up3 = UpBlock(c[3], c[2])                   # /4
        self.up4 = UpBlock(c[2], c[1])                   # /2
        self.up5 = UpBlock(c[1], c[0])                   # /1

        self.outc = nn.Conv1d(c[0], out_channels, 1)

    def forward(self, x):
        x0 = self.inc(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.bottleneck(x5)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)

        return self.outc(x)


if __name__ == '__main__':
    net = RealUNet()
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Real U-Net params: {n / 1e6:.2f} M")
    x = torch.randn(4, 2, 8192)
    y = net(x)
    print(f"{x.shape} → {y.shape}")
