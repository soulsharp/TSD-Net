import torch

from torch import nn
from model.TSD_model.TSD_blocks import CTEBlock, CPSABlock, LFFNBlock
from utils.utils import load_yaml

class TSD_Encoder(nn.Module):
    def __init__(self, dim, num_heads, expansion_ratio):
        super(TSD_Encoder, self).__init__()
        self.cpsa_block = CPSABlock(dim, num_heads)
        self.lffn_block = LFFNBlock(dim, kernel_size=3, expansion_ratio=expansion_ratio)
    
    def forward(self, x, h, w):
        x = self.cpsa_block(x ,h, w)
        # _, C, H, W = x.shape
        x = self.lffn_block(x, h, w)
        
        return x

class TSD_T(nn.Module):
    def __init__(self, dim, cte_out_channels, num_heads=2, num_encoder_layers=2, expansion_ratio=2):
        super(TSD_T, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.cte_block = CTEBlock(cte_out_channels, dim)
        self.encoder_layers = nn.ModuleList([
            TSD_Encoder(dim, num_heads=num_heads, expansion_ratio=expansion_ratio) for _ in range(num_encoder_layers)
        ])
    
    def forward(self, x):
        x, h, w = self.cte_block(x)
        for encoder in self.encoder_layers:
            x = encoder(x, h, w)

        return x

class TSD_B(nn.Module):
    def __init__(self, dim, cte_out_channels, num_heads=4, num_encoder_layers=6, expansion_ratio=4):
        super(TSD_B, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.cte_block = CTEBlock(cte_out_channels, dim)
        self.encoder_layers = nn.ModuleList([
            TSD_Encoder(dim, num_heads=num_heads, expansion_ratio=expansion_ratio) for _ in range(num_encoder_layers)
        ])
    
    def forward(self, x):
        x, h, w = self.cte_block(x)
        for encoder in self.encoder_layers:
            x = encoder(x, h, w)

        return x