import torch
from torch import nn, Tensor
from torch.nn import Conv1d
from torch.nn import functional as F
import numpy as np
import math
from typing import Union
#from mamba_ssm import Mamba

def ConvBlock(
    in_channels,
    out_channels,
    kernel_size,
    padding="same",
    stride=1,
    dilation=1,
    bias=True,
    batch_norm=True,
    activ_funct="relu",
):
    if activ_funct == "relu":
        activation = nn.ReLU(inplace=True)
    if activ_funct == "gelu":
        activation = nn.GELU(inplace=True)
    return nn.Sequential(
        (
            nn.BatchNorm1d(in_channels) if batch_norm else nn.Identity()
        ),  
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        ),
        activation,
    )

class Permute21(nn.Module):
    def __init__(self):
        super(Permute21, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

def MambaBlock(n_filters, d_state=16, d_conv=3, expand=2):
    return nn.Sequential(
        nn.Sequential(
            Permute21(),
            Mamba(
                d_model=n_filters, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand,
            ),
            Permute21(),
        )
    )

def TransformerBlock(n_filters, nhead=3, expand=2, n_layers=1):
    return nn.Sequential(
        nn.Sequential(
            Permute21(),
            nn.LayerNorm(n_filters),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=n_filters, 
                    nhead=nhead, 
                    dim_feedforward=n_filters*expand,
                    batch_first=True,
                ),
                num_layers=n_layers,
            ),
            Permute21(),
        )
    )

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._module = module
    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.dim = dim

        self.to_q = nn.Linear(dim, dim * heads, bias=False)
        self.to_k = nn.Linear(dim, dim * heads, bias=False)
        self.to_v = nn.Linear(dim, dim * heads, bias=False)
        self.to_out = nn.Linear(dim * heads, dim)

    def forward(self, x, y):
        b, _, _ = x.shape
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)

        q = self.to_q(x).view(b, -1, self.heads, x.size(-1)).transpose(1, 2)
        k = self.to_k(y).view(b, -1, self.heads, y.size(-1)).transpose(1, 2)
        v = self.to_v(y).view(b, -1, self.heads, y.size(-1)).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, self.heads * x.size(-1))

        return self.to_out(out)