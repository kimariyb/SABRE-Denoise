import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities import rank_zero_warn

from network.sabre_encoder import Transformer
from network.sabre_decoder import DecoderCup, SegmentationHead


# TODO: 实现 SabreNet
class SabreNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SabreNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

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