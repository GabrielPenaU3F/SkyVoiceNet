import torch.nn as nn

from source.network.convolutional_layer import ConvolutionalEncoderLayer, ConvolutionalDecoderLayer

class ConvolutionalBlock(nn.Module):

    def __init__(self, layers):
        super(ConvolutionalBlock, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvolutionalEncoderBlock(ConvolutionalBlock):

    def __init__(self, in_channels, out_channels):
        layer_1 = ConvolutionalEncoderLayer(in_channels, out_channels=32, kernel_size=(3, 3), stride=(3, 3),
                                            padding=(1, 1))
        layer_2 = ConvolutionalEncoderLayer(in_channels=32, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layers = [layer_1, layer_2]
        '''
        For a 4-layer architecture
            
        layer_1 = ConvolutionalEncoderLayer(in_channels, out_channels=16, kernel_size=(3, 3), stride=(3, 3),
                                            padding=(1, 1))
        layer_2 = ConvolutionalEncoderLayer(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layer_3 = ConvolutionalEncoderLayer(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layer_4 = ConvolutionalEncoderLayer(in_channels=64, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=(1, 1), padding=(1, 1))
        layers = [layer_1, layer_2, layer_3, layer_4]
        '''
        super().__init__(layers)


class ConvolutionalDecoderBlock(ConvolutionalBlock):

    def __init__(self, in_channels, out_channels):
        layer_1 = ConvolutionalDecoderLayer(in_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layer_2 = ConvolutionalDecoderLayer(in_channels=32, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=(3, 3), padding=(1, 1), output_padding=(2, 0))
        layers = [layer_1, layer_2]
        '''
        For a 4-layer architecture
        layer_1 = ConvolutionalDecoderLayer(in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layer_2 = ConvolutionalDecoderLayer(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layer_3 = ConvolutionalDecoderLayer(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1))
        layer_4 = ConvolutionalDecoderLayer(in_channels=16, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=(3, 3), padding=(1, 1), output_padding=(2, 0))
        layers = [layer_1, layer_2, layer_3, layer_4]
        '''
        super().__init__(layers)
