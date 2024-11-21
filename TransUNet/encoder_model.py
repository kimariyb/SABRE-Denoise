import math
import copy

import torch
import torch.nn as nn

from einops import rearrange

from TransUNet.conv_model import ResNet


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embedding_dim: int,
        dropout: float,
    ):
        super(PatchEmbedding, self).__init__()
        
        self.seq_length = 256
        self.in_channels = 128
        self.patch_size = patch_size
        self.patch_num = self.seq_length // self.patch_size 

        self.embedding_dim = embedding_dim
        
        self.resnet = ResNet()
        
        self.patch_embeddings = nn.Conv1d(self.in_channels, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_num, self.embedding_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x, features = self.resnet(x) # (B, C , L)
        x = self.patch_embeddings(x) # (B, C, L) -> (B, D, N)
        x = x.transpose(1, 2) # (B, D, N) -> (B, N, D)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, features


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        num_heads: int,
        embedding_dim: int,
        attn_dropout: float,
    ):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        self.query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value =nn. Linear(self.embedding_dim, self.embedding_dim)

        self.out = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

        self.act = nn.GELU()
    
    def forward(self, x):
        q = rearrange(
            self.query(x), "b n (h d) -> b h n d", h=self.num_heads
        ) 
        k = rearrange(
            self.key(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        v = rearrange(
            self.value(x), "b n (h d) -> b h n d", h=self.num_heads
        )

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = self.act(attention_scores)
        weights = attention_probs 
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embedding_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights
    
    
class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        dropout: float,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        ffn_embedding_dim,
        num_heads,
        dropout,
        attn_dropout
    ):
        super(TransformerBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn = MLP(embedding_dim, ffn_embedding_dim, dropout)
        self.attn = MultiHeadAttention(num_heads, embedding_dim, attn_dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        return x, weights

            
class TransUNetEncoder(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        ffn_embedding_dim,
        num_heads,
        num_layers,
        dropout,
        attn_dropout,
    ):
        super(TransUNetEncoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        for _ in range(num_layers):
            layer = TransformerBlock(embedding_dim, ffn_embedding_dim, num_heads, dropout, attn_dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        attn_weights = []
        
        for layer_block in self.layer:
            x, weights = layer_block(x)
            attn_weights.append(weights)
                
        encoded = self.encoder_norm(x)
        
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        ffn_embedding_dim,
        num_heads,
        num_layers,
        patch_size,
        dropout,
        attn_dropout,
    ):
        super(Transformer, self).__init__()
        
        self.embeddings = PatchEmbedding(
            embedding_dim=embedding_dim, 
            patch_size=patch_size,
            dropout=dropout,
        )
         
        self.encoder = TransUNetEncoder(
            embedding_dim=embedding_dim, 
            ffn_embedding_dim=ffn_embedding_dim, 
            num_heads=num_heads, 
            num_layers=num_layers,
            dropout=dropout, 
            attn_dropout=attn_dropout, 
         )

    def forward(self, x):  
        x, features = self.embeddings(x)
        
        encoded, attn_weights = self.encoder(x)  # (B, n_patch, hidden)
        
        return encoded, attn_weights, features
    
