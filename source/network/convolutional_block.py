import torch.nn as nn

from source.network.residual_buffer import ResidualBuffer


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Conv1DBlock, self).__init__()

        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm = nn.BatchNorm1d(out_channels)
        activation = nn.LeakyReLU()

        self.block = nn.Sequential(conv, norm, activation)

    def forward(self, x):
        return self.block(x)


class ConvTranspose1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0):
        super(ConvTranspose1DBlock, self).__init__()

        conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, output_padding=output_padding, bias=False)
        norm = nn.BatchNorm1d(out_channels)
        activation = nn.LeakyReLU()

        self.block = nn.Sequential(conv, norm, activation)

    def forward(self, x):
        return self.block(x)
