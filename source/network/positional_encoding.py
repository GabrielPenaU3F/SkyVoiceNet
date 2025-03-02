import torch
from torch import nn

from source.config import PositionalEncodingConfig


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.config = PositionalEncodingConfig()
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding formulae
        pe = torch.zeros(max_len, d_model).to(self.config.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] * 0.1
        return self.dropout(x)
