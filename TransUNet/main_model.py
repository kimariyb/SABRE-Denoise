import re

import torch
import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_warn

from TransUNet.encoder_model import Transformer
from TransUNet.decoder_model import DecoderCup, UpsamplingBilinear1d


class SpectralDeNoiser(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = UpsamplingBilinear1d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv1d, upsampling)

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# TODO: 实现 SabreNet
class SabreNet(nn.Module):
    def __init__(
        self,  
        vis,
        seq_length, 
        in_channels,
        embedding_dim, 
        ffn_embedding_dim, 
        num_heads, 
        num_layers, 
        patches, 
        dropout, 
        attn_dropout, 
        decoder_channels, 
        n_skip,
        skip_channels,
    ):
        super(SabreNet, self).__init__()
        self.transformer = Transformer(
            vis=vis, seq_length=seq_length, in_channels=in_channels, embedding_dim=embedding_dim, 
            ffn_embedding_dim=ffn_embedding_dim, num_heads=num_heads, num_layers=num_layers, patches=patches, dropout=dropout, attn_dropout=attn_dropout)
        self.decoder = DecoderCup(
            embedding_dim=embedding_dim, decoder_channels=decoder_channels, n_skip=n_skip, skip_channels=skip_channels
        )
        self.denoiser = SpectralDeNoiser(
            in_channels=decoder_channels,
            out_channels=2,
        )

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.denoiser(x)
        
        return logits    
    
    
def create_model(
        vis,
        seq_length, 
        in_channels,
        embedding_dim, 
        ffn_embedding_dim, 
        num_heads, 
        num_layers, 
        patches, 
        dropout, 
        attn_dropout, 
        decoder_channels, 
        n_skip,
        skip_channels,
    ):
    return SabreNet(
        vis=vis,
        seq_length=seq_length,
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        ffn_embedding_dim=ffn_embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        patches=patches,
        dropout=dropout,
        attn_dropout=attn_dropout,
        decoder_channels=decoder_channels,
        n_skip=n_skip,
        skip_channels=skip_channels,
    )


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")

    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            rank_zero_warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    
    return model.to(device)