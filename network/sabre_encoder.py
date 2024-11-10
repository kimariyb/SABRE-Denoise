import math
import copy

import torch
import torch.nn as nn

from network.sabre_resnet_skip import ResNet


class Attention(nn.Module):
    def __init__(
        self, 
        vis,
        num_heads,
        embedding_dim,
        attn_dropout,
    ):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_heads = num_heads
        self.head_dim = int(embedding_dim / self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(embedding_dim, self.all_head_size)
        self.key = nn.Linear(embedding_dim, self.all_head_size)
        self.value =nn. Linear(embedding_dim, self.all_head_size)

        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)
    
    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights
    
    
class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        dropout_rate,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.act_fn = nn.GELU()
        self.dropout = nn.Linear(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    
class Embeddings(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        dropout_rate,
        patches,
        input_size,
        resnet, 
        in_channels=2
    ):
        super(Embeddings, self).__init__()
        
        self.hybrid = None

        if patches.get("grid") is not None:   # ResNet
            grid_size = patches["grid"]
            patch_size = (input_size[0] // 16 // grid_size[0], input_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (input_size[0] // patch_size_real[0]) * (input_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = patches["size"]
            n_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNet(block_units=resnet.num_layers, width_factor=resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
            
        self.patch_embeddings = nn.Conv1d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, embedding_dim))

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
            
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings, features
    
    
class Block(nn.Module):
    def __init__(
        self, 
        vis,
        embedding_dim, 
        ffn_embedding_dim,
        num_heads,
        dropout_rate,
        attn_dropout
    ):
        super(Block, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn = MLP(embedding_dim, ffn_embedding_dim, dropout_rate)
        self.attn = Attention(vis, num_heads, embedding_dim, attn_dropout)

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

            
class Encoder(nn.Module):
    def __init__(
        self, 
        vis,
        embedding_dim,
        ffn_embedding_dim,
        num_heads,
        dropout_rate,
        attn_dropout,
        num_layers,
    ):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(vis, embedding_dim, ffn_embedding_dim, num_heads, dropout_rate, attn_dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(
        self, 
        vis, 
        embedding_dim,
        ffn_embedding_dim,
        num_heads,
        dropout_rate,
        attn_dropout,
        num_layers,
        patches,
        input_size,
        resnet,
        in_channels=2
    ):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(embedding_dim, dropout_rate, patches, input_size, resnet, in_channels)
        self.encoder = Encoder(vis, embedding_dim, ffn_embedding_dim, num_heads, dropout_rate, attn_dropout, num_layers)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features