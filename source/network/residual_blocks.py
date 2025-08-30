from torch import nn

from source.config import NetworkConfig


class ResidualBlock(nn.Module):

    def __init__(self, in_freqs, hidden_dim=32):
        super(ResidualBlock, self).__init__()
        self.config = NetworkConfig()
        self.conv = nn.Conv1d(in_freqs, hidden_dim, 5, stride=1, padding=2)
        self.recurrent = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.transp_conv = nn.ConvTranspose1d(hidden_dim, in_freqs, 5, padding=2)

    def forward(self, x):
        y = self.conv(x)
        y, _ = self.recurrent(y.permute(0, 2, 1))
        y = self.transp_conv(y.permute(0, 2, 1))
        y = x + y
        return y
