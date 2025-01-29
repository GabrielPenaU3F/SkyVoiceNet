import torch.nn as nn

from source.network.convolutional_layer import ConvolutionalLayer


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()
        self.layer_1 = ConvolutionalLayer(in_channels, out_channels=32,
                                          kernel_size=(5, 5), stride=(3, 3), padding=(1, 1))
        self.layer_2 = ConvolutionalLayer(in_channels=32, out_channels=out_channels,
                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x
