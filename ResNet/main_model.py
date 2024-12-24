import torch.nn as nn

from ResNet.encoder_model import ResNet
from ResNet.decoder_model import DecoderCup


class SabreNet(nn.Module):
    def __init__(self):
        super(SabreNet, self).__init__()
        self.encoder = ResNet()
        self.decoder = DecoderCup()

    def forward(self, x):
        x, features = self.encoder(x)  # (B, n_patch, hidden)
        logits = self.decoder(x, features)
        
        return logits    

    
def create_model():
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
    model = SabreNet()
    
    # Initialize weights and parameters
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    return model





