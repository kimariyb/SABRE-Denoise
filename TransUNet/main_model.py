import re

import torch
import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_warn

from TransUNet.encoder_model import Transformer
from TransUNet.decoder_model import DecoderCup, UpsamplingBilinear1d


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

class SpectralDeNoiser(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super(SpectralDeNoiser, self).__init__()        
        # 定义卷积层
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # 定义上采样层
        self.upsampling = UpsamplingBilinear1d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # 定义激活函数
        self.swish = Swish()

        # 初始化权重
        self.init_weights()

    def forward(self, x):
        # 前向传播
        x = self.conv1d(x)
        x = self.upsampling(x)
        x = self.swish(x)
        return x
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.zeros_(self.conv1d.bias)
        if hasattr(self.upsampling, "weight"):
            nn.init.xavier_uniform_(self.upsampling.weight)
        if hasattr(self.upsampling, "bias"):
            nn.init.zeros_(self.upsampling.bias)


class SabreNet(nn.Module):
    def __init__(
        self,  
        vis: bool,
        seq_length: int, 
        in_channels: int,
        embedding_dim: int, 
        ffn_embedding_dim: int, 
        num_heads: int, 
        num_layers: int, 
        patch_size: int, 
        dropout: float, 
        attn_dropout: float, 
        decoder_channels: list, 
        skip_channels: list,
        skip_num: int,
    ):
        super(SabreNet, self).__init__()
        self.transformer = Transformer(
            vis=vis, seq_length=seq_length, in_channels=in_channels, embedding_dim=embedding_dim, 
            ffn_embedding_dim=ffn_embedding_dim, num_heads=num_heads, num_layers=num_layers, 
            patch_size=patch_size, dropout=dropout, attn_dropout=attn_dropout
        )
        self.decoder = DecoderCup(
            embedding_dim=64, decoder_channels=decoder_channels, skip_channels=skip_channels, skip_num=skip_num
        )
        self.denoiser = SpectralDeNoiser(
            in_channels=decoder_channels[-1],
            out_channels=2,
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.transformer.init_weights()
        self.decoder.init_weights()
        self.denoiser.init_weights()

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.denoiser(x)
        
        return logits    

    
def create_model(args):
    """
    Create a model from the given args.
    
    Parameters
    ----------
    args : dict
        Dictionary of arguments.
    
    Returns
    -------
    model : nn.Module
        The created model.
    """
    net_args = dict(
        vis=args['vis'],
        seq_length=args['seq_length'],
        in_channels=args['in_channels'],
        embedding_dim=args['embedding_dim'],
        ffn_embedding_dim=args['ffn_embedding_dim'],
        num_heads=args['num_heads'],
        num_layers=args['num_layers'],
        patch_size=args['patch_size'],
        dropout=args['dropout'],
        attn_dropout=args['attn_dropout'],
        skip_num=args['skip_num'],
        decoder_channels=args['decoder_channels'],
        skip_channels=args['skip_channels'],
    )
    
    model = SabreNet(**net_args)
    
    return model


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