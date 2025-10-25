"""
Complete MSFM + ViT model

Author: Syed Mohd Tashif
Email: syedtashif239@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
from .msfm import CompleteMSFM
from .vit import ViTEncoder


class CompleteMSFMViTModel(nn.Module):
    def __init__(self, num_classes=20, embed_dim=512):
        super().__init__()
        self.msfm = CompleteMSFM(out_channels=embed_dim)
        self.vit = ViTEncoder(num_layers=4, embed_dim=embed_dim, num_heads=8, mlp_dim=1024)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        msfm_out = self.msfm(x)
        vit_out = self.vit(msfm_out)
        pooled = vit_out.mean(dim=1)
        return torch.sigmoid(self.head(pooled))
