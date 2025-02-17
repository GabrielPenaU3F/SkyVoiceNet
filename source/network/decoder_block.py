from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_block import ConvolutionalDecoderBlock
from source.network.transformer_block import TransformerBlock


class DecoderBlock(nn.Module):

    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.transformer_block = TransformerBlock(
            self.config.transf_dim, num_heads=self.config.transf_heads, hidden_dim=self.config.transf_hidden,
            num_layers=self.config.transf_num_layers, dropout=self.config.transf_dropout, device=self.config.device)
        self.conv_input_projection = None
        self.conv_block = ConvolutionalDecoderBlock(in_channels=self.config.conv_out_channels, out_channels=1)



    def forward(self, attended_audio):
        """
            Advice:
            There will be some ifs to dynamically define layers,
            this is because we can't know a-priori the dimensions of certain outputs
        """

        transformer_output = self.transformer_block(attended_audio)

        # Dynamically define the projection layer, only once
        if self.conv_input_projection is None:
            self.conv_input_projection = nn.Linear(
                self.config.transf_dim, self.config.conv_output_dim).to(transformer_output.device)

        # Project and reformat
        projected_output = self.conv_input_projection(transformer_output)
        conv_input = self.format_convolutional_input(projected_output)

        # Apply the convolutional block to recover a reconstructed spectrogram
        output_spectrogram = self.conv_block(conv_input)

        return output_spectrogram


    # This is the exact inverse operation of the formatting done in the encoder
    def format_convolutional_input(self, transf_output):
        batch_size, time, features = transf_output.shape
        channels = self.config.conv_out_channels
        freq = features // channels
        conv_input = transf_output.view(batch_size, time, channels, freq)
        return conv_input.permute(0, 2, 3, 1).contiguous()  # [batch, channels, freq, time]
