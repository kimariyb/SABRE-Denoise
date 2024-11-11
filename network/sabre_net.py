import re

import torch
import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_warn

from network.sabre_encoder import Transformer
from network.sabre_decoder import DecoderCup, SegmentationHead


# TODO: 实现 SabreNet
class SabreNet(nn.Module):
    def __init__(
        self,  
        embedding_dim, 
        ffn_embedding_dim, 
        num_heads, 
        num_layers, 
        in_size, 
        in_channels,
        patches, 
        dropout, 
        attn_dropout, 
        resnet,
        decoder_channels, 
        n_skip,
        skip_channels,
        num_classes, 
        zero_head, 
        vis
    ):
        super(SabreNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.transformer = Transformer(
            vis, embedding_dim, ffn_embedding_dim, ...)
        self.decoder = DecoderCup(embedding_dim, decoder_channels, n_skip, skip_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits    
    
def create_model(in_channels=2, out_channels=2):
    return SabreNet(in_channels=in_channels, out_channels=out_channels)

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