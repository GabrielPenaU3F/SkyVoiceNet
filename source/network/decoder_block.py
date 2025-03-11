from torch import nn
from torch.nn import functional as F

from source.config import NetworkConfig
from source.network.convolutional_block import ConvolutionalDecoderBlock
from source.network.residual_buffer import ResidualBuffer
from source.network.transformer_decoder_block import TransformerDecoderBlock


class DecoderBlock(nn.Module):

    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.buffer = ResidualBuffer()
        self.config.update(**kwargs)
        self.blstm = nn.LSTM(input_size=self.config.transf_embedding_dim, hidden_size=128,
                             num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        # self.transformer_decoder_block = TransformerDecoderBlock(
        #     self.config.transf_embedding_dim, num_heads=self.config.transf_heads, hidden_dim=self.config.transf_hidden,
        #     num_layers=self.config.transf_num_layers, dropout=self.config.transf_dropout)
        self.conv_block = ConvolutionalDecoderBlock(in_channels=self.config.conv_out_channels, out_channels=1)


    def forward(self, attended_audio):

        # Transformer
        # Option 1: Transformer
        # transformer_output = self.transformer_decoder_block(attended_audio)

        # Option 2: BLSTM
        transformer_output, _ = self.blstm(attended_audio)

        # Option 3: Bypass
        # transformer_output = attended_audio

        # Skip connections
        # residual = self.buffer.retrieve_transformer_input_buffer()
        # transformer_output = transformer_output + residual
        # transformer_output = transformer_output + attended_audio

        # Reformat
        transformer_output = transformer_output.unsqueeze(1)
        transformer_output = F.interpolate(
            transformer_output, size=(transformer_output.shape[-2], self.config.conv_features_dim), mode='bilinear')  # Ajuste de tamaño
        conv_input = self.format_convolutional_input(transformer_output.squeeze(1))

        # Apply the convolutional block to recover a reconstructed spectrogram
        conv_output = self.conv_block(conv_input)

        # Map the output to the correct size
        output_spectrogram = F.interpolate(conv_output, size=self.config.spectrogram_dimensions, mode='bilinear')  # Ajuste de tamaño

        return output_spectrogram

    # This is the exact inverse operation of the formatting done in the encoder
    def format_convolutional_input(self, transf_output):
        batch_size, time, features = transf_output.shape
        channels = self.config.conv_out_channels
        freq = features // channels
        conv_input = transf_output.view(batch_size, time, channels, freq)
        return conv_input.permute(0, 2, 3, 1).contiguous()  # [batch, channels, freq, time]
