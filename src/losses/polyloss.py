"""
Poly-1 BCE Loss implementation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyBCELoss(nn.Module):
    def __init__(self, epsilon=1.0, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        poly_loss = bce_loss + self.epsilon * (1 - pt)
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss
