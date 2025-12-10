import torch
from torch import nn
from torch.nn import functional as F

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(LearnedPositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.positional_encoding.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention and add & norm
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward and add & norm
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, d_model, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # self.pos_encoder = LearnedPositionalEncoding(seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, src):
        # Add positional encoding
        # src = self.pos_encoder(src)

        # Pass through each Transformer layer
        for layer in self.layers:
            src = layer(src)

        return src
    
