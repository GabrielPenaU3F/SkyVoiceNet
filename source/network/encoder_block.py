import torch
from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_block import ConvolutionalEncoderBlock
from source.network.transformer_encoder_block import TransformerEncoderBlock


class EncoderBlock(nn.Module):

    def __init__(self, **kwargs):
        super(EncoderBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.conv_block = ConvolutionalEncoderBlock(in_channels=1, out_channels=self.config.conv_out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4096, 256))
        self.blstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        # self.transformer_encoder_block = TransformerEncoderBlock(
        #     self.config.transf_embedding_dim, num_heads=self.config.transf_heads, hidden_dim=self.config.transf_hidden,
        #     num_layers=self.config.transf_num_layers, dropout=self.config.transf_dropout)

    def forward(self, speech_spec):

        # Save this for later use in the decoder
        self.config.spectrogram_dimensions = speech_spec.shape[-2:]  # Guardar (H, W) originales

        # Convolutional leyers and feed-forward to produce a correct transformer input
        conv_output = self.conv_block(speech_spec)
        transformer_input = self.format_transformer_input(conv_output)
        self.config.conv_output_dim = transformer_input.shape[-1] # Save this for later use in the decoder

        # Transformer layer
        transformer_input = self.global_avg_pool(transformer_input)
        # Option 1: Transformer
        # speech_embedding = self.transformer_encoder_block(transformer_input)
        # Option 2: BLSTM
        speech_embedding, _ = self.blstm(transformer_input)
        # Option 3: Bypass
        # speech_embedding = transformer_input

        return speech_embedding

    def format_transformer_input(self, conv_output):
        batch_size, channels, freq, time = conv_output.shape
        conv_output = conv_output.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, freq]
        return conv_output.view(batch_size, time, -1)  # [batch, time, features]
