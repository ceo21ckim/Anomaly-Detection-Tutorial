import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import pandas as pd 
import math 


    
class AnomalyAttention(nn.Module):
    def __init__(self, window_size):
        super(AnomalyAttention, self).__init__()
        self.window_size = window_size 
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)
    
    def forward(self, queries, keys, values, sigma):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape 
        scale = 1. / math.sqrt(E) # d_model # scaled dot product
        
        scores = torch.einsum('blhe, bshe->bhls', queries, keys) # matmul Queries and Keys
        
        attn = scale * scores 
        
        sigma = sigma.transpose(1, 2) # B L H -> B H L 
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size) # B H L L 
        
        
        # prior association
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1. / (math.sqrt(2 * math.pi) * sigma ) * torch.exp(-prior ** 2 / 2/ (sigma ** 2) )
        
        
        # series association
        series = torch.softmax(attn, dim=-1)
        V = torch.einsum('bhls, bshd->blhd', series, values) # reconstruction
        
        return V.contiguous(), series, prior, sigma
    
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        
        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)
        
        self.norm = nn.LayerNorm(d_model)
        self.inner_projection = attention
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape 
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries 
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        
        out, series, prior, sigma = self.inner_projection(
            queries, keys, values, sigma
        )
        out = out.view(B, L, -1)
        
        return self.out_projection(out), series, prior, sigma 

    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        
        self.PE = torch.zeros(max_len, d_model).float().cuda()

        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * - (math.log(10000.0) / d_model)).exp()
        
        self.PE[:, 0::2] = torch.sin(position * div_term)
        self.PE[:, 1::2] = torch.cos(position * div_term)
        
        self.PE = self.PE.unsqueeze(0)
        self.register_buffer('pe', self.PE)
        
    def forward(self, x):
        return self.PE[:, :x.size(1)]
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular', 
            bias=False
        )
        
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x 
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()
        
        self.value_emb = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_emb = PositionalEmbedding(d_model=d_model)
        
    def forward(self, x):
        x = self.value_emb(x) + self.position_emb(x)
        return x 
    
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None):
        super(EncoderLayer,self).__init__()
        d_ff = 4 * d_model 
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x): # attention 
        new_x, attn, mask, sigma = self.attention(
            x, x, x)
        x = x + new_x 
        y = self.norm1(x)
        y = self.activation(self.conv1(y.transpose(-1,1))) # Feed Forward Networks
        y = self.conv2(y).transpose(-1, 1)
        return self.norm2(x + y), attn, mask, sigma
    
class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        
    def forward(self, x):
        series_list, prior_list, sigma_list = [], [], []
        
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            
        
        return x, series_list, prior_list, sigma_list 
    
class AnomalyTransformer(nn.Module):
    def __init__(self, window_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512):
        super(AnomalyTransformer, self).__init__()
        self.output_attention=True 
        
        self.embedding = DataEmbedding(enc_in, d_model)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(window_size), 
                        d_model, n_heads), 
                    d_model, 
                    d_ff
                ) for l in range(e_layers)
            ], 
        )
        
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigma = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        
        return enc_out, series, prior, sigma 
                
        