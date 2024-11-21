import torch.nn as nn

from TransUNet.encoder_model import Transformer
from TransUNet.decoder_model import DecoderCup


class SabreNet(nn.Module):
    def __init__(
        self,  
        embedding_dim: int, 
        ffn_embedding_dim: int, 
        num_heads: int, 
        num_layers: int, 
        patch_size: int, 
        dropout: float, 
        attn_dropout: float, 
    ):
        super(SabreNet, self).__init__()
        self.transformer = Transformer(
            embedding_dim=embedding_dim, ffn_embedding_dim=ffn_embedding_dim, num_heads=num_heads, num_layers=num_layers, 
            patch_size=patch_size, dropout=dropout, attn_dropout=attn_dropout
        )
        self.decoder = DecoderCup()
        

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        logits = self.decoder(x, features)
        
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
        embedding_dim=args['embedding_dim'],
        ffn_embedding_dim=args['ffn_embedding_dim'],
        num_heads=args['num_heads'],
        num_layers=args['num_layers'],
        patch_size=args['patch_size'],
        dropout=args['dropout'],
        attn_dropout=args['attn_dropout'],
    )
    
    model = SabreNet(**net_args)
    
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





