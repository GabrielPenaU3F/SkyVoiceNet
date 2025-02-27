import torch.nn as nn

from source.network.positional_encoding import PositionalEncoding


class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout, device):
        super(TransformerEncoderBlock, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim, dropout, device=device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return x
