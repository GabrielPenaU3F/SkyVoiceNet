import torch
from torch import nn
from torch.nn import functional as F

from source.config import NetworkConfig
from source.network.positional_encoding import PositionalEncoding


class AttentionBlock(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.positional_encoding = PositionalEncoding(d_model=self.config.embed_dim, max_len=5000)
        # self.melody_dim_reduction = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=0),
        #     nn.GroupNorm(num_groups=8, num_channels=32),
        #     nn.SiLU(),
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.GroupNorm(num_groups=8, num_channels=64),
        #     nn.SiLU(),
        #     nn.InstanceNorm2d(self.config.conv_out_channels)
        # )
        # self.pool = nn.AdaptiveAvgPool1d(self.config.transf_embedding_dim)
        self.attention = nn.MultiheadAttention(self.config.embed_dim, num_heads=1,
                                               dropout=self.config.cross_attention_dropout, batch_first=True)

    def forward(self, speech_embedding, melody_contour, speech_spec):

        # For the version with convolutional dimension reduction
        # melody_contour = self.format_contour_pre_conv(melody_contour)
        # melody_embedding = self.melody_dim_reduction(melody_contour) # Convolutional reduction
        # melody_embedding = self.format_contour_post_conv(melody_embedding, batch_size)
        # melody_embedding = self.pool(melody_embedding)

        # Alternatively, the simpler version with just permutations and downsampling
        melody_contour, speech_spec = self.format_inputs(melody_contour, speech_spec)

        # Positional encode
        # speech_embedding = self.positional_encoding(speech_embedding)
        # melody_contour = self.positional_encoding(melody_contour)

        attention_output, attn_output_weights = self.attention(query=speech_embedding,
                                                          key=melody_contour,
                                                          value=speech_spec)

        # Residual connection
        # attention_output = attention_output + speech_embedding

        # Concatenation?
        # combined_embedding = torch.cat([speech_embedding, melody_embedding], dim=-1)
        # attention_output = self.projection(combined_embedding)

        return attention_output


    def format_inputs(self, melody_contour, speech_spec):
        melody_contour = melody_contour.to(self.config.device)
        time = melody_contour.shape[-1]
        melody_contour = F.interpolate(melody_contour, size=(time, self.config.embed_dim), mode='bilinear')
        speech_spec = F.interpolate(speech_spec, size=(time, self.config.embed_dim), mode='bilinear')
        melody_contour = melody_contour.squeeze(1)
        speech_spec = speech_spec.squeeze(1)
        return melody_contour, speech_spec


    # def format_contour_pre_conv(self, melody_contour):
    #     melody_contour = melody_contour.to(self.config.device)
    #     batch_size, channels, num_freqs, time = melody_contour.shape  # Will be the 128 MIDIs in this implementation
    #     melody_contour = melody_contour.view(batch_size * num_freqs, channels, time)  # [batch * freq, channels, time]
    #     return melody_contour
    #
    # def format_contour_post_conv(self, melody_embedding, batch_size):
    #     batch_freqs, channels, time = melody_embedding.shape
    #     num_freqs =  batch_freqs // batch_size
    #     melody_embedding = melody_embedding.view(batch_size, num_freqs, channels, time)  # [batch, freq, out_channels, time]
    #     melody_embedding = melody_embedding.view(batch_size, time, num_freqs * channels) # [batch, embeddings, time]
    #     return melody_embedding
