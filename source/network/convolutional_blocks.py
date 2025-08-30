import torch.nn as nn

from source.config import NetworkConfig


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Conv1DBlock, self).__init__()
        self.config = NetworkConfig()

        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm = nn.BatchNorm1d(out_channels, eps=1e-6)
        activation = nn.LeakyReLU()
        dropout_layer = nn.Dropout1d(self.config.dropout)
        self.block = nn.Sequential(conv, norm, activation, dropout_layer)

    def forward(self, x):
        return self.block(x)


class ConvTranspose1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0):
        super(ConvTranspose1DBlock, self).__init__()
        self.config = NetworkConfig()

        conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, output_padding=output_padding, bias=False)
        norm = nn.BatchNorm1d(out_channels, eps=1e-6)
        activation = nn.LeakyReLU()
        dropout_layer = nn.Dropout1d(self.config.dropout)
        self.block = nn.Sequential(conv, norm, activation, dropout_layer)

    def forward(self, x):
        return self.block(x)
