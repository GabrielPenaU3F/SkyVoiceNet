import torch
from torch import nn


class SoftClip(nn.Module):

    def forward(self, x):
        return (x / (1 + torch.abs(x)) + 1) / 2  # Escalado a [0,1]


class BoundedSwish(nn.Module):

    def __init__(self, beta=1.0, eps=1e-6):
        """
        BoundedSwish con saturación ajustable.
        - beta < 1: más saturación (mayor contraste).
        - beta > 1: menos saturación (más suave).
        """
        super().__init__()
        self.eps = eps
        self.beta = beta  # Parámetro de saturación ajustable

    def forward(self, x):
        # Swish ajustado con beta
        swish = (x * torch.sigmoid(self.beta * x)) / self.beta
        return (swish - swish.min()) / (swish.max() - swish.min() + self.eps)



class NormalizedTanh(nn.Module):

    def forward(self, x):
        return (1 + torch.tanh(x)) / 2