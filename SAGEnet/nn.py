import torch
from torch import nn, Tensor
from torch.nn import Conv1d
from torch.nn import functional as F
import numpy as np
import math


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
        activation = nn.GELU()
    elif activ_funct == "exp":
        activation = ExpActivation()  
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
