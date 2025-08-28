import torch
from torch import nn

import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        half_dim = self.dim // 2
        emb = math.log(1000) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * (-emb))
        emb = x.unsqueeze(-1) * emb.view(([1] * x.ndim) + [-1])
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1).float()
        return emb