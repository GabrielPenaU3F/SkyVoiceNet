import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConvolutionalLayer(nn.Module):

    def __init__(self, out_channels):
        super(ConvolutionalLayer, self).__init__()
        self.conv = None
        self.norm = nn.BatchNorm2d(out_channels)
        # self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.silu(x)
        x = self.norm(x)
        return x



class ConvolutionalEncoderLayer(ConvolutionalLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)


class ConvolutionalDecoderLayer(ConvolutionalLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 output_padding=(0, 0)):
        super().__init__(out_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)
