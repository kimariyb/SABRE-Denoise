from typing import List, Optional

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """SE-Net 风格通道注意力，输入通道数为实际通道数（2C）"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.gap(x).squeeze(-1)  # [B, 2C]
        w = self.fc(w).unsqueeze(-1)  # [B, 2C, 1]
        return x * w


class SpectralAttention(nn.Module):
    """位置注意力，关注峰区域"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class CBAM1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpectralAttention(channels)

    def forward(self, x):
        return self.sa(self.ca(x))


class MultiScaleBlock(nn.Module):
    """
    空洞卷积多尺度聚合。
    输入 [B, 2C, W]，整体当作 2C 通道特征处理（不拆分复数）。
    """
    def __init__(self, channels: int):
        """channels = 实际张量通道数，即 2 × 复数通道数"""
        super().__init__()
        assert channels % 4 == 0
        c4 = channels // 4
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, c4, 1),
            nn.Conv1d(channels, c4, 3, padding=1),
            nn.Conv1d(channels, c4, 3, padding=4, dilation=4),
            nn.Conv1d(channels, c4, 3, padding=8, dilation=8),
        ])
        self.fuse = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.PReLU()

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.act(self.bn(self.fuse(out))) + x


class GatedSkip(nn.Module):
    def __init__(self, channels: int):
        """channels = 实际张量通道数（2C）"""
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, decoder_feat, encoder_feat):
        g = self.gate(torch.cat([decoder_feat, encoder_feat], dim=1))
        return encoder_feat * g


class CConv1d(nn.Module):
    """
    复数 Conv1d
    in_ch / out_ch 均指复数通道数（张量实际通道 = 2×）
    (a+jb)*(c+jd) = (ac-bd) + j(ad+bc)
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, activation: bool = True):
        super().__init__()
        self.real_conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.imag_conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn_real = nn.BatchNorm1d(out_ch)
        self.bn_imag = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU() if activation else None

        nn.init.kaiming_normal_(self.real_conv.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.imag_conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.chunk(2, dim=1)
        real_out = self.bn_real(self.real_conv(real) - self.imag_conv(imag))
        imag_out = self.bn_imag(self.real_conv(imag) + self.imag_conv(real))
        out = torch.cat([real_out, imag_out], dim=1)  # [B, 2*out_ch, W']
        return self.act(out) if self.act is not None else out


class CConvTranspose1d(nn.Module):
    """
    复数 ConvTranspose1d
    in_ch / out_ch 均指复数通道数（张量实际通道 = 2×）
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 4, stride: int = 2,
                 padding: int = 1, output_padding: int = 0,
                 activation: bool = True):
        super().__init__()
        self.real_tconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                                             stride, padding, output_padding)
        self.imag_tconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                                             stride, padding, output_padding)
        self.bn_real = nn.BatchNorm1d(out_ch)
        self.bn_imag = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.chunk(2, dim=1)
        real_out = self.bn_real(self.real_tconv(real) - self.imag_tconv(imag))
        imag_out = self.bn_imag(self.real_tconv(imag) + self.imag_tconv(real))
        out = torch.cat([real_out, imag_out], dim=1)  # [B, 2*out_ch, W']
        return self.act(out) if self.act is not None else out


class SDNBlock(nn.Module):
    def __init__(self, in_channels: int = 1,  # 复数通道数
                 channels: Optional[List[int]] = None,
                 out_channels: int = 1,  # 复数通道数
                 use_attention: bool = True,
                 use_gated_skip: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.use_gated_skip = use_gated_skip

        # self.c 存复数通道数；实际张量通道 = 2 × c
        self.c = [16, 32, 64, 128, 256, 512] if channels is None else channels
        assert len(self.c) == 6

        # ── 初始层：1 → c[0] 复数通道，不改变长度 ──
        self.init_layer = CConv1d(in_channels, self.c[0], 3, 1, 1)

        # ── 编码器：stride=2 下采样 ──
        self.encoder = nn.ModuleList([
            CConv1d(self.c[0], self.c[1], 4, 2, 1),
            CConv1d(self.c[1], self.c[2], 4, 2, 1),
            CConv1d(self.c[2], self.c[3], 4, 2, 1),
            CConv1d(self.c[3], self.c[4], 4, 2, 1),
            CConv1d(self.c[4], self.c[5], 4, 2, 1),
        ])

        # ── 编码器注意力（作用于 2×c[i] 实际通道）──
        if self.use_attention:
            self.enc_attn = nn.ModuleList([
                CBAM1d(self.c[i + 1] * 2) for i in range(5)
            ])

        # ── 瓶颈：多尺度（作用于 2×c[5] 实际通道）──
        self.bottleneck = MultiScaleBlock(self.c[5] * 2)

        # ── 门控 skip（通道数均为实际通道 2×c[...]）──
        if self.use_gated_skip:
            self.gated_skips = nn.ModuleList([
                GatedSkip(self.c[5 - 1 - i] * 2) for i in range(4)
            ])

        # ── 解码器：stride=2 上采样 ──
        self.decoder = nn.ModuleList([
            CConvTranspose1d(self.c[5], self.c[4], 4, 2, 1, 0),
            CConvTranspose1d(self.c[4] * 2, self.c[3], 4, 2, 1, 0),
            CConvTranspose1d(self.c[3] * 2, self.c[2], 4, 2, 1, 0),
            CConvTranspose1d(self.c[2] * 2, self.c[1], 4, 2, 1, 0),
            CConvTranspose1d(self.c[1] * 2, self.c[0], 4, 2, 1, 0),
        ])

        # ── 输出层 ──
        self.out_layer = CConv1d(self.c[0], out_channels, 3, 1, 1, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2*in_channels, W]
        x = self.init_layer(x)  # [B, 2*c[0], W]

        # ── 编码 ──
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.use_attention:
                x = self.enc_attn[i](x)
            skips.append(x)  # 实际通道 2*c[i+1]

        # ── 瓶颈 ──
        x = self.bottleneck(x)  # [B, 2*c[5], W/32]

        # ── 解码 ──
        x = self.decoder[0](x)  # 第一层无 skip

        for i, layer in enumerate(self.decoder[1:]):
            skip = skips[-(i + 2)]
            if self.use_gated_skip:
                skip = self.gated_skips[i](x, skip)
            x = torch.cat([x, skip], dim=1)  # 复数 cat：2*c + 2*c = 4*c（实际通道）
            x = layer(x)

        return self.out_layer(x)  # [B, 2*out_channels, W]


class SabreSDN(nn.Module):
    """
    输入/输出：[B, 2, W]，通道 0 = 实部，通道 1 = 虚部
    内部以 1 个复数通道进入 SDNBlock（in_channels=1）
    """
    def __init__(self, num_blocks: int = 1,
                 use_attention: bool = True,
                 use_gated_skip: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SDNBlock(in_channels=1, out_channels=1,
                     use_attention=use_attention,
                     use_gated_skip=use_gated_skip)
            for _ in range(num_blocks)
        ])
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, W]  — 已经是 [实部, 虚部] 格式
        init_x = x
        for block in self.blocks:
            x = block(x)
            x = x + init_x  # 全局残差
            init_x = x
        return self.tanh(x)


if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = SabreSDN(num_blocks=1, use_attention=True, use_gated_skip=True).to(device)

    total = sum(p.numel() for p in net.parameters())
    print(f"参数量: {total / 1e6:.2f} M")

    x = torch.randn(4, 2, 8192).to(device)
    with torch.no_grad():
        y = net(x)
    print(f"输入: {x.shape}  →  输出: {y.shape}")
    assert x.shape == y.shape, "形状不匹配！"