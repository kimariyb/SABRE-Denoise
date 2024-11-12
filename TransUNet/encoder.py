import math
import copy

import torch
import torch.nn as nn

from resnet import ResNet


class Attention(nn.Module):
    def __init__(
        self, 
        vis: bool,
        num_heads: int,
        embedding_dim: int,
        attn_dropout: float,
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
        
        self.reset_parameters()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.normal_(self.query.bias, std=1e-6)
        nn.init.normal_(self.key.bias, std=1e-6)
        nn.init.normal_(self.value.bias, std=1e-6)
        nn.init.normal_(self.out.bias, std=1e-6)    
    
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
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
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
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
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
        dropout,
        seq_length,
        patch_size,
        in_channels,
        block_units, 
    ):
        super(Embeddings, self).__init__()
                
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.n_patches = self.seq_length // self.patch_size
        self.in_channels = in_channels

        self.resnet = ResNet(block_units=block_units, length=seq_length)
            
        self.patch_embeddings = nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.embedding_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.embedding_dim))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x, features = self.resnet(x)
   
        # (batch_size, in_channels, seq_length) -> (batch_size, hidden_size, n_patches)
        x = self.patch_embeddings(x) 
        x = x.transpose(1, 2)  # (B, n_patches, hidden_size)
        
        # Flatten the patches, (B, n_patches, hidden_size) -> (B, n_patches * hidden_size)
        x = x.reshape(x.size(0), x.size(2), -1) 
        
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
        dropout,
        attn_dropout
    ):
        super(Block, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.ffn = MLP(embedding_dim, ffn_embedding_dim, dropout)
        self.attn = Attention(vis, num_heads, embedding_dim, attn_dropout)
        
    def init_weights(self):
        self.ffn.reset_parameters()
        self.attn.reset_parameters()

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
        num_layers,
        dropout,
        attn_dropout,
    ):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(vis, embedding_dim, ffn_embedding_dim, num_heads, dropout, attn_dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        attn_weights = []
        
        for layer_block in self.layer:
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
                
        encoded = self.encoder_norm(x)
        
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(
        self, 
        vis, 
        embedding_dim,
        ffn_embedding_dim,
        num_heads,
        num_layers,
        patch_size,
        seq_length,
        dropout,
        attn_dropout,
        in_channels,
        block_units,
    ):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(
            embedding_dim=embedding_dim, 
            dropout=dropout,
            seq_length=seq_length,
            patch_size=patch_size,
            in_channels=in_channels,
            block_units=block_units,
        )
         
        self.encoder = Encoder(
            vis=vis, 
            embedding_dim=embedding_dim, 
            ffn_embedding_dim=ffn_embedding_dim, 
            num_heads=num_heads, 
            num_layers=num_layers,
            dropout=dropout, 
            attn_dropout=attn_dropout, 
         )

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        
        return encoded, attn_weights, features
    

class TestTransformer:
    def __init__(self):
        # 设置超参数
        self.vis = False
        self.embedding_dim = 768
        self.ffn_embedding_dim = 3072
        self.num_heads = 12
        self.num_layers = 6
        self.patch_size = 16
        self.seq_length = 8192
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.in_channels = 3  # 假设输入是 RGB 图像
        self.block_units = [2, 2, 2, 2]  # 每个层块的单位数
        
        # 创建 Transformer 实例
        self.model = Transformer(
            vis=self.vis,
            embedding_dim=self.embedding_dim,
            ffn_embedding_dim=self.ffn_embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            patch_size=self.patch_size,
            seq_length=self.seq_length,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            in_channels=self.in_channels,
            block_units=self.block_units,
        )
    
    def test_forward(self):
        # 创建一个随机输入张量 (batch_size, channels, length)
        batch_size = 1
        input_tensor = torch.randn(batch_size, self.in_channels, self.seq_length)

        # 前向传播
        encoded, attn_weights, features = self.model(input_tensor)

        # 打印结果形状
        print(f"Encoded shape: {encoded.shape}")
        print(f"Attention weights shape: {len(attn_weights)}")
        if attn_weights:
            print(f"Attention weights first layer shape: {attn_weights[0].shape}")
        print(f"Features shape: {features.shape if features is not None else 'None'}")

if __name__ == "__main__":
    tester = TestTransformer()
    tester.test_forward()
