import torch.nn as nn

from source.network.positional_encoding import PositionalEncoding


class TransformerDecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout, device):
        super(TransformerDecoderBlock, self).__init__()

        self.positional_encoding = PositionalEncoding(input_dim, dropout, device=device)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


    def forward(self, x, memory, mask=None):
        x = self.positional_encoding(x)
        output = self.transformer_decoder(x, memory, tgt_mask=mask)
        return output