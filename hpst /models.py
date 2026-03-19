"""Graph Neural Network architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn_graph(coords, k=10):
    """Build k-NN graph."""
    dist = torch.cdist(coords, coords)
    knn = dist.topk(k+1, largest=False).indices[:, 1:]
    src = torch.arange(len(coords), device=coords.device).repeat_interleave(k)
    dst = knn.reshape(-1)
    return torch.stack([src, dst])

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.scale = (dim // heads) ** 0.5
    
    def forward(self, x, edge_index):
        N, src, dst = x.size(0), edge_index[0], edge_index[1]
        q, k, v = self.qkv(x).view(N, self.heads, self.head_dim * 3).chunk(3, dim=-1)
        
        attn = (q[src] * k[dst]).sum(dim=-1) / self.scale
        exp_attn = torch.exp(attn - attn.max(dim=0, keepdim=True)[0])
        
        sum_exp = torch.zeros(N, self.heads, device=x.device)
        sum_exp.index_add_(0, dst, exp_attn)
        attn_weights = exp_attn / (sum_exp[dst] + 1e-8)
        
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        out.index_add_(0, dst, v[dst] * attn_weights.unsqueeze(-1))
        return self.out(out.view(N, -1))

class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
    
    def forward(self, x, edge_index):
        x = x + self.attn(self.norm1(x), edge_index)
        x = x + self.ffn(self.norm2(x))
        return x

class GraphTransformer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=2, 
                 n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.embed(x)
        for layer in self.layers:
            x = self.dropout(layer(x, edge_index))
        return self.out(self.norm(x))

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=2, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers-1):
            layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, *args):
        return self.net(x)
