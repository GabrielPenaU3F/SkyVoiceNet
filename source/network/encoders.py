from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_blocks import Conv1DBlock
from source.network.residual_buffer import ResidualBuffer


class Encoder(nn.Module):

    def __init__(self, freqs=512, hidden_dim=64):
        super(Encoder, self).__init__()
        self.config = NetworkConfig()
        self.residual_buffer = ResidualBuffer()

        # Dimensional reduction
        hidden_1 = int(freqs / 2)
        hidden_2 = int(freqs / 4)

        # Each convolutional channel represents a frequency bin

        # Stride 1 ensures keeping temporal dimension constant
        self.f_downsample_conv_1 = Conv1DBlock(freqs, hidden_1, kernel_size=5, stride=1, padding=2)
        self.f_downsample_conv_2 = Conv1DBlock(hidden_1, hidden_2, kernel_size=5, stride=1, padding=2)
        self.f_downsample_conv_3 = Conv1DBlock(hidden_2, hidden_dim, kernel_size=5, stride=1, padding=2)

        # We keep channels constant and use stride 2 to reduce the temporal dimension
        self.t_downsample_conv_1 = Conv1DBlock(freqs, freqs, kernel_size=5, stride=2, padding=2)
        self.t_downsample_conv_2 = Conv1DBlock(hidden_1, hidden_1, kernel_size=5, stride=2, padding=2)
        self.t_downsample_conv_3 = Conv1DBlock(hidden_2, hidden_2, kernel_size=5, stride=2, padding=2)

        # LSTM or other recurrent layers
        self.norm = nn.InstanceNorm1d(hidden_2)
        self.lstm = nn.LSTM(hidden_2, hidden_2, num_layers=2, batch_first=True, dropout=self.config.dropout, bidirectional=True)
        self.lstm_proj = nn.Conv1d(hidden_2 * 2, hidden_2, kernel_size=5, stride=1, padding=2)


    def forward(self, x):

        x = x.squeeze(1)

        # Downsample up to a dimension of 128

        # Downsampling 1
        x_down2_t = self.t_downsample_conv_1(x)
        x_down2 = self.f_downsample_conv_1(x_down2_t)
        self.residual_buffer.buffer_conv_1_output(x_down2)

        # Downsampling 2
        x_down4_t = self.t_downsample_conv_2(x_down2)
        x_down4 = self.f_downsample_conv_2(x_down4_t)
        self.residual_buffer.buffer_conv_2_output(x_down4)

        # Apply LSTM when the embedding dimension is 128
        x_down4 = self.norm(x_down4)
        x_down4, _ = self.lstm(x_down4.permute(0, 2, 1))
        x_down4 = x_down4.permute(0, 2, 1)
        x_down4 = self.lstm_proj(x_down4)

        # Downsampling 3
        x_down8_t = self.t_downsample_conv_3(x_down4)
        x_down8 = self.f_downsample_conv_3(x_down8_t)
        self.residual_buffer.buffer_conv_3_output(x_down8)

        encoding = x_down8
        return encoding
