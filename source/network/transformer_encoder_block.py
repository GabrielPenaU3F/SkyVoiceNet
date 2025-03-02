import torch.nn as nn

from source.network.positional_encoding import PositionalEncoding


class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim, max_len=5000, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.initialize_transformer()

    def initialize_transformer(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        x = self.positional_encoding(x)
        # for layer in self.transformer_encoder.layers:
        #     print(f"Antes de LayerNorm: mean={x.mean()}, std={x.std()}") # 🔍 Check values
        #     x = layer(x)
        #     print(f"Después de LayerNorm: mean={x.mean()}, std={x.std()}")  # 🔍 Check if it shrinks too much
        x = self.transformer_encoder(x)
        return x
