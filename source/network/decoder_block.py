import torch
from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_block import ConvolutionalDecoderBlock
from source.network.transformer_decoder_block import TransformerDecoderBlock


class DecoderBlock(nn.Module):

    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.fc = None
        self.transformer_decoder_block = TransformerDecoderBlock(
            self.config.transf_embedding_dim, num_heads=self.config.transf_heads, hidden_dim=self.config.transf_hidden,
            num_layers=self.config.transf_num_layers, dropout=self.config.transf_dropout, device=self.config.device)
        self.conv_block = ConvolutionalDecoderBlock(in_channels=self.config.conv_out_channels, out_channels=1)


    def forward(self, attended_audio, memory):

        # Transformer
        transformer_output = self.transformer_decoder_block(attended_audio, memory=memory)
        print("Transformer Decoder Output:", torch.mean(transformer_output), torch.std(transformer_output))

        # Define projection layer
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(self.config.transf_embedding_dim, self.config.conv_output_dim),
                nn.LayerNorm(self.config.conv_output_dim),
                nn.LeakyReLU()
            ).to(self.config.device)

        # Project and reformat
        projected_output = self.fc(transformer_output)
        conv_input = self.format_convolutional_input(projected_output)

        # Apply the convolutional block to recover a reconstructed spectrogram
        output_spectrogram = self.conv_block(conv_input)

        print("Final Spectrogram Output:", torch.mean(output_spectrogram), torch.std(output_spectrogram))

        return output_spectrogram


    # This is the exact inverse operation of the formatting done in the encoder
    def format_convolutional_input(self, transf_output):
        batch_size, time, features = transf_output.shape
        channels = self.config.conv_out_channels
        freq = features // channels
        conv_input = transf_output.view(batch_size, time, channels, freq)
        return conv_input.permute(0, 2, 3, 1).contiguous()  # [batch, channels, freq, time]
