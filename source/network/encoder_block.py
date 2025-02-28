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
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=self.config.transf_embedding_dim, device=self.config.device),
            nn.LayerNorm(self.config.transf_embedding_dim, device=self.config.device),
            nn.LeakyReLU()
        )
        self.transformer_encoder_block = TransformerEncoderBlock(
            self.config.transf_embedding_dim, num_heads=self.config.transf_heads, hidden_dim=self.config.transf_hidden,
            num_layers=self.config.transf_num_layers, dropout=self.config.transf_dropout, device=self.config.device)

    def forward(self, speech_spec):

        # Convolutional leyers and feed-forward to produce a correct transformer input
        conv_output = self.conv_block(speech_spec)
        transf_input = self.format_transformer_input(conv_output)
        self.config.conv_output_dim = transf_input.shape[-1] # Save this for later use in the decoder
        print("Convolutional encoder output:", torch.mean(conv_output), torch.std(conv_output))

        # Transformer layer
        transformer_input = self.fc(transf_input)
        speech_embedding = self.transformer_encoder_block(transformer_input)
        print("Speech Embedding (Transformer Encoder Output):", torch.mean(speech_embedding),
              torch.std(speech_embedding))
        return speech_embedding

    def format_transformer_input(self, conv_output):
        batch_size, channels, freq, time = conv_output.shape
        conv_output = conv_output.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, freq]
        return conv_output.view(batch_size, time, -1)  # [batch, time, features]
