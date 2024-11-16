import re

import torch
import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_warn

from encoder_model import Transformer
from decoder_model import DecoderCup, UpsamplingBilinear1d


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
    ):
        super(SabreNet, self).__init__()
        self.transformer = Transformer(
            vis=vis, seq_length=seq_length, in_channels=in_channels, embedding_dim=embedding_dim, 
            ffn_embedding_dim=ffn_embedding_dim, num_heads=num_heads, num_layers=num_layers, 
            patch_size=patch_size, dropout=dropout, attn_dropout=attn_dropout
        )
        self.decoder = DecoderCup(
            embedding_dim=64, decoder_channels=decoder_channels, skip_channels=skip_channels
        )
        # self.denoiser = SpectralDeNoiser(
        #     in_channels=decoder_channels,
        #     out_channels=2,
        # )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.transformer.init_weights()
        self.decoder.init_weights()
        # self.denoiser.init_weights()

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        # logits = self.denoiser(x)
        
        return x    


class TestSabreNet:
    def __init__(self):
        # 初始化参数
        self.vis = True
        self.seq_length = 512
        self.in_channels = 64
        self.embedding_dim = 2048
        self.ffn_embedding_dim = 4096
        self.num_heads = 16
        self.num_layers = 12
        self.patch_size = 32
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.decoder_channels = [256, 128, 64, 16]
        self.skip_channels = [32, 16, 8, 4]

        # 创建SabreNet的实例
        self.model = SabreNet(
            vis=self.vis,
            seq_length=self.seq_length,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            ffn_embedding_dim=self.ffn_embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            patch_size=self.patch_size,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            decoder_channels=self.decoder_channels,
            skip_channels=self.skip_channels
        )

    def run_test(self):
        # 创建一个随机输入张量
        input_tensor = torch.randn(1, 2, 8192)  
        output = self.model(input_tensor)

        # 打印输出形状
        print("输出形状:", output.shape)

# 执行测试
if __name__ == "__main__":
    test_sabre_net = TestSabreNet()
    test_sabre_net.run_test()


    
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
        patches=args['patches'],
        dropout=args['dropout'],
        attn_dropout=args['attn_dropout'],
        n_skip=args['n_skip'],
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