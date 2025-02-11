from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_block import ConvolutionalBlock
from source.network.transformer_block import TransformerBlock


class SpeechEncoderBlock(nn.Module):

    def __init__(self, **kwargs):
        super(SpeechEncoderBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.conv_block = ConvolutionalBlock(in_channels=1, out_channels=self.config.conv_out_channels)
        self.conv_output_projection = None
        self.transformer_block = TransformerBlock(
            self.config.transf_dim, num_heads=self.config.transf_heads, hidden_dim=self.config.transf_hidden,
            num_layers=self.config.transf_num_layers, dropout=self.config.transf_dropout, device=self.config.device)

    def forward(self, speech_spec):
        """
            Advice:
            There will be some ifs to dynamically define layers,
            this is because we can't know a-priori the dimensions of certain outputs
        """

        # Convolutional leyers and feed-forward to produce a correct transformer input
        conv_output = self.conv_block(speech_spec)
        transf_input = self.format_transformer_input(conv_output)

        # Dynamically define the projection layer, only once
        if self.conv_output_projection is None:
            self.conv_output_projection = nn.Linear(
                transf_input.shape[-1], self.config.transf_dim).to(speech_spec.device)

        # Transformer layer
        transformer_input = self.conv_output_projection(transf_input)
        speech_embedding = self.transformer_block(transformer_input)
        return speech_embedding

    def format_transformer_input(self, conv_output):
        batch_size, channels, freq, time = conv_output.shape
        conv_output = conv_output.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, freq]
        return conv_output.view(batch_size, time, -1)  # [batch, time, features]
