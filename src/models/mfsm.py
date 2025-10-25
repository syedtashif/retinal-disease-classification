"""
Multi-Scale Feature Map (MSFM) module

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CompleteMSFM(nn.Module):
    """Complete MSFM architecture with DenseNet-201 backbone"""

    def __init__(self, out_channels=512):
        super().__init__()
        densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        self.basemodel = densenet.features
        self.basemodel_l = nn.Sequential(
            self.basemodel.conv0, self.basemodel.norm0, self.basemodel.relu0, self.basemodel.pool0,
            self.basemodel.denseblock1, self.basemodel.transition1,
            self.basemodel.denseblock2, self.basemodel.transition2
        )
        self.high_conv = nn.Conv2d(1920, out_channels, 1)
        self.low_conv = nn.Conv2d(256, out_channels, 1)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        self.fusion = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cam_conv1 = nn.Conv2d(out_channels, out_channels, 1)
        self.cam_conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.mha = nn.MultiheadAttention(embed_dim=out_channels, num_heads=8, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        f_high = self.basemodel(x)
        f_low = self.basemodel_l(x)
        f_dash_h = self.high_conv(f_high)
        f_l = self.low_conv(f_low)
        f_u = self.upsample(f_dash_h)
        if f_u.shape != f_l.shape:
            f_u = F.interpolate(f_u, size=f_l.shape[-2:], mode='bilinear', align_corners=False)
        f_c = f_u + f_l
        f_dash_l = self.fusion(f_c)
        f_p = self.gap(f_dash_l)
        f_v = F.relu(self.cam_conv1(f_p))
        f_v = torch.sigmoid(self.cam_conv2(f_v))
        f_v_attended = f_dash_l * f_v
        f_ddash_l = f_dash_l + f_v_attended
        Q = f_dash_h.flatten(2).permute(0, 2, 1)
        K = f_ddash_l.flatten(2).permute(0, 2, 1)
        V = f_ddash_l.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.mha(Q, K, V)
        skip_x_attention = self.norm1(Q + attn_out)
        ffn_out = self.ffn(skip_x_attention)
        ffn_out = self.dropout(ffn_out)
        msfm_out = self.norm2(skip_x_attention + ffn_out)
        return msfm_out
