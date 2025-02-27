import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConvolutionalLayer(nn.Module):

    def __init__(self, out_channels):
        super(ConvolutionalLayer, self).__init__()
        self.conv = None
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)


class ConvolutionalEncoderLayer(ConvolutionalLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.apply(self.init_weights)


class ConvolutionalDecoderLayer(ConvolutionalLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 output_padding=(0, 0)):
        super().__init__(out_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.apply(self.init_weights)
