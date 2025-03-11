import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from source.network.convolutional_layer import ConvolutionalEncoderLayer, ConvolutionalDecoderLayer
from source.network.residual_buffer import ResidualBuffer


class ConvolutionalBlock(nn.Module):

    def __init__(self, layer_1, layer_2):
        super(ConvolutionalBlock, self).__init__()
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.buffer = ResidualBuffer()  # Singleton


class ConvolutionalEncoderBlock(ConvolutionalBlock):

    def __init__(self, in_channels, out_channels):
        layer_1 = ConvolutionalEncoderLayer(in_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                            padding=(0, 0))
        layer_2 = ConvolutionalEncoderLayer(in_channels=32, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(0, 0))
        super().__init__(layer_1, layer_2)

    def forward(self, x):
        x = self.layer_1(x)
        self.buffer.buffer_conv_1_output(x) # Skip connection 1
        x = self.layer_2(x)
        self.buffer.buffer_conv_2_output(x) # Skip connection 2
        return x


class ConvolutionalDecoderBlock(ConvolutionalBlock):

    def __init__(self, in_channels, out_channels):
        layer_1 = ConvolutionalDecoderLayer(in_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(0, 0), output_padding=(0, 0))
        layer_2 = ConvolutionalDecoderLayer(in_channels=32, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=(2, 2), padding=(0, 0), output_padding=(0, 0))
        super().__init__(layer_1, layer_2)

    def forward(self, x):
        buffer_1 = self.buffer.retrieve_buffer_conv_1_output()
        buffer_2 = self.buffer.retrieve_buffer_conv_2_output()
        x = self.layer_1(x + buffer_2)
        x = self.layer_2(x + buffer_1)
        return x
