"""
Ablation model variants for SabreSDN component analysis.

Provides 7 architecture variants by toggling individual components:
  1. Full SabreSDN        — all components enabled
  2. − CConv → Real Conv  — CConv1d/CConvTranspose1d → standard Conv1d/ConvTranspose1d
  3. − CBAM1d (both)      — remove both channel attention and spectral attention
  4. − Channel Attn only  — remove channel attention, keep spectral attention
  5. − Spectral Attn only — remove spectral attention, keep channel attention
  6. − MultiScale → Single— MultiScaleBlock → single Conv1d 3×1
  7. − GatedSkip → Concat — sigmoid gating → plain concatenation

(8. − Composite Loss → L1 is handled by the training script, not the model.)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn

from models import (
    CConv1d, CConvTranspose1d, CBAM1d,
    ChannelAttention, SpectralAttention,
    MultiScaleBlock, GatedSkip,
)


# ── Real-valued conv replacements ────────────────────────────────────────
def _real_conv(in_ch, out_ch, kernel, stride, padding, activation=True):
    """Standard Conv1d to replace one CConv1d layer."""
    layers = [nn.Conv1d(in_ch, out_ch, kernel, stride, padding),
              nn.BatchNorm1d(out_ch)]
    if activation:
        layers.append(nn.PReLU())
    return nn.Sequential(*layers)


def _real_tconv(in_ch, out_ch, kernel, stride, padding,
                output_padding=0, activation=True):
    """Standard ConvTranspose1d to replace one CConvTranspose1d layer."""
    layers = [nn.ConvTranspose1d(in_ch, out_ch, kernel, stride, padding,
                                  output_padding),
              nn.BatchNorm1d(out_ch)]
    if activation:
        layers.append(nn.PReLU())
    return nn.Sequential(*layers)


# ── Single-scale bottleneck (replaces MultiScaleBlock) ───────────────────
class SingleScaleBlock(nn.Module):
    """Single Conv1d 3×1 + BN + PReLU + residual.  Replaces MultiScaleBlock."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + x


# ── Ablation attention wrapper ───────────────────────────────────────────
class AblationAttention(nn.Module):
    """Attention module supporting full / CA-only / SA-only."""
    def __init__(self, channels: int, use_ca: bool = True, use_sa: bool = True,
                 reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction) if use_ca else None
        self.sa = SpectralAttention(channels) if use_sa else None

    def forward(self, x):
        if self.ca is not None:
            x = self.ca(x)
        if self.sa is not None:
            x = self.sa(x)
        return x


# ── Ablation SDNBlock ────────────────────────────────────────────────────
class AblationSDNBlock(nn.Module):
    """
    SDNBlock with component toggles for ablation.

    Args:
        use_cconv:         use CConv1d/CConvTranspose1d (True) or standard Conv1d (False)
        use_channel_attn:  use ChannelAttention in encoder
        use_spectral_attn: use SpectralAttention in encoder
        use_multiscale:    use MultiScaleBlock (True) or SingleScaleBlock (False)
        use_gated_skip:    use GatedSkip (True) or plain concat (False)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 use_cconv: bool = True,
                 use_channel_attn: bool = True,
                 use_spectral_attn: bool = True,
                 use_multiscale: bool = True,
                 use_gated_skip: bool = True):
        super().__init__()
        self.use_cconv = use_cconv
        self.use_gated_skip = use_gated_skip
        self.use_channel_attn = use_channel_attn
        self.use_spectral_attn = use_spectral_attn
        self.use_multiscale = use_multiscale

        # complex channels for CConv; tensor channels = 2×
        self.cc = [16, 32, 64, 128, 256, 512]
        self.tc = [2 * c for c in self.cc]  # [32, 64, 128, 256, 512, 1024]

        # ── Build layers ─────────────────────────────────────────────
        in_tc = in_channels * 2   # = 2
        out_tc = out_channels * 2  # = 2

        if use_cconv:
            self.init_layer = CConv1d(in_channels, self.cc[0], 3, 1, 1)
            self.encoder = nn.ModuleList([
                CConv1d(self.cc[0], self.cc[1], 4, 2, 1),
                CConv1d(self.cc[1], self.cc[2], 4, 2, 1),
                CConv1d(self.cc[2], self.cc[3], 4, 2, 1),
                CConv1d(self.cc[3], self.cc[4], 4, 2, 1),
                CConv1d(self.cc[4], self.cc[5], 4, 2, 1),
            ])
            # Decoder: first layer (no skip) then 4 layers with skip concat
            self.decoder = nn.ModuleList([
                CConvTranspose1d(self.cc[5], self.cc[4], 4, 2, 1, 0),
                CConvTranspose1d(self.cc[4] * 2, self.cc[3], 4, 2, 1, 0),
                CConvTranspose1d(self.cc[3] * 2, self.cc[2], 4, 2, 1, 0),
                CConvTranspose1d(self.cc[2] * 2, self.cc[1], 4, 2, 1, 0),
                CConvTranspose1d(self.cc[1] * 2, self.cc[0], 4, 2, 1, 0),
            ])
            self.out_layer = CConv1d(self.cc[0], out_channels, 3, 1, 1,
                                     activation=False)
        else:
            # Real-valued: directly use tensor channel counts
            self.init_layer = _real_conv(in_tc, self.tc[0], 3, 1, 1)
            self.encoder = nn.ModuleList([
                _real_conv(self.tc[0], self.tc[1], 4, 2, 1),
                _real_conv(self.tc[1], self.tc[2], 4, 2, 1),
                _real_conv(self.tc[2], self.tc[3], 4, 2, 1),
                _real_conv(self.tc[3], self.tc[4], 4, 2, 1),
                _real_conv(self.tc[4], self.tc[5], 4, 2, 1),
            ])
            self.decoder = nn.ModuleList([
                _real_tconv(self.tc[5], self.tc[4], 4, 2, 1),
                _real_tconv(self.tc[4] * 2, self.tc[3], 4, 2, 1),
                _real_tconv(self.tc[3] * 2, self.tc[2], 4, 2, 1),
                _real_tconv(self.tc[2] * 2, self.tc[1], 4, 2, 1),
                _real_tconv(self.tc[1] * 2, self.tc[0], 4, 2, 1),
            ])
            self.out_layer = nn.Conv1d(self.tc[0], out_tc, 3, 1, 1)

        # Attention
        use_any_attn = use_channel_attn or use_spectral_attn
        if use_any_attn:
            self.enc_attn = nn.ModuleList([
                AblationAttention(self.tc[i], use_channel_attn, use_spectral_attn)
                for i in range(1, 6)  # t[1]=64 .. t[5]=1024
            ])

        # Bottleneck
        if use_multiscale:
            self.bottleneck = MultiScaleBlock(self.tc[5])
        else:
            self.bottleneck = SingleScaleBlock(self.tc[5])

        # Gated skip
        if use_gated_skip:
            self.gated_skips = nn.ModuleList([
                GatedSkip(self.tc[4 - i]) for i in range(4)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, W]
        x = self.init_layer(x)

        # Encoder
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if (self.use_channel_attn or self.use_spectral_attn):
                x = self.enc_attn[i](x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder[0](x)   # first layer, no skip
        for i, layer in enumerate(self.decoder[1:]):
            skip = skips[-(i + 2)]
            if self.use_gated_skip:
                skip = self.gated_skips[i](x, skip)
            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        return self.out_layer(x)


class AblationSabreSDN(nn.Module):
    """SabreSDN wrapper for ablation study."""
    def __init__(self, num_blocks: int = 1, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            AblationSDNBlock(in_channels=1, out_channels=1, **kwargs)
            for _ in range(num_blocks)
        ])
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_x = x
        for block in self.blocks:
            x = block(x) + init_x
            init_x = x
        return self.tanh(x)


# ── Factory ──────────────────────────────────────────────────────────────
VARIANTS = {
    'full': {
        'use_cconv': True,
        'use_channel_attn': True,
        'use_spectral_attn': True,
        'use_multiscale': True,
        'use_gated_skip': True,
    },
    'no_cconv': {
        'use_cconv': False,
        'use_channel_attn': True,
        'use_spectral_attn': True,
        'use_multiscale': True,
        'use_gated_skip': True,
    },
    'no_cbam': {
        'use_cconv': True,
        'use_channel_attn': False,
        'use_spectral_attn': False,
        'use_multiscale': True,
        'use_gated_skip': True,
    },
    'ca_only': {  # − Spectral Attention, keep Channel Attention
        'use_cconv': True,
        'use_channel_attn': True,
        'use_spectral_attn': False,
        'use_multiscale': True,
        'use_gated_skip': True,
    },
    'sa_only': {  # − Channel Attention, keep Spectral Attention
        'use_cconv': True,
        'use_channel_attn': False,
        'use_spectral_attn': True,
        'use_multiscale': True,
        'use_gated_skip': True,
    },
    'no_multiscale': {
        'use_cconv': True,
        'use_channel_attn': True,
        'use_spectral_attn': True,
        'use_multiscale': False,
        'use_gated_skip': True,
    },
    'no_gated_skip': {
        'use_cconv': True,
        'use_channel_attn': True,
        'use_spectral_attn': True,
        'use_multiscale': True,
        'use_gated_skip': False,
    },
}

# Aliases matching the paper terminology
VARIANT_NAMES = {
    'full':             'Full SabreSDN',
    'no_cconv':         '- CConv1d -> Real Conv',
    'no_cbam':          '- CBAM1d (both)',
    'ca_only':          '- Spectral Attention only',
    'sa_only':          '- Channel Attention only',
    'no_multiscale':    '- MultiScale -> Single',
    'no_gated_skip':    '- GatedSkip -> Concat',
}


def build_ablation_model(variant: str) -> AblationSabreSDN:
    """Build a SabreSDN variant by key.

    Args:
        variant: one of 'full', 'no_cconv', 'no_cbam', 'ca_only',
                 'sa_only', 'no_multiscale', 'no_gated_skip'

    Returns:
        AblationSabreSDN instance
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Choose from: {list(VARIANTS.keys())}")
    return AblationSabreSDN(**VARIANTS[variant])


if __name__ == '__main__':
    for key in VARIANTS:
        net = build_ablation_model(key)
        n = sum(p.numel() for p in net.parameters() if p.requires_grad)
        x = torch.randn(2, 2, 8192)
        with torch.no_grad():
            y = net(x)
        print(f"{VARIANT_NAMES[key]:<35s}  {n/1e6:.1f}M  {tuple(x.shape)} -> {tuple(y.shape)}")
