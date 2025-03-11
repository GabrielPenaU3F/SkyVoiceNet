from torch import nn

from source.config import NetworkConfig
from source.network.cross_attention_layer import CrossAttentionLayer
from source.network.positional_encoding import PositionalEncoding


class AttentionBlock(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.positional_encoding = PositionalEncoding(d_model=self.config.transf_embedding_dim, max_len=5000)
        self.cross_attention = CrossAttentionLayer(self.config.transf_embedding_dim, self.config.cross_attention_num_heads,
                                                   dropout=self.config.cross_attention_dropout)

    def forward(self, speech_embedding, melody_contour):

        # Format contour and project
        melody_contour = self.format_contour(melody_contour)
        melody_contour = self.fc(melody_contour)

        # Positional encode
        speech_embedding = self.positional_encoding(speech_embedding)
        melody_contour = self.positional_encoding(melody_contour)

        # Attend
        attention_output, attn_weights = self.cross_attention(speech_embedding, melody_contour)

        return attention_output

    def format_contour(self, melody_contour):
        batch_size, _, num_freqs, _ = melody_contour.shape  # Will be the 128 MIDIs in this implementation
        melody_contour = melody_contour.view(batch_size, num_freqs, -1).permute(0, 2, 1)  # [batch, time, frequencies]
        return melody_contour
