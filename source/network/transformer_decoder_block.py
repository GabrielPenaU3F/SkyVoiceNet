import torch.nn as nn

from source.network.positional_encoding import PositionalEncoding
from source.network.residual_buffer import ResidualBuffer


class TransformerDecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.buffer = ResidualBuffer()
        self.positional_encoding = PositionalEncoding(input_dim, max_len=600, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.initialize_transformer()

    def initialize_transformer(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x, mask=None):
        x_residual = x
        x = self.positional_encoding(x)
        memory = self.buffer.retrieve_transformer_output_buffer()
        x = self.transformer_decoder(x, memory, tgt_mask=mask)
        x = x + x_residual
        return x
