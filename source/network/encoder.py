import torch
from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_block import Conv1DBlock


class Encoder(nn.Module):
    """ Encoder de Speech o Melodía basado en convoluciones 1D """
    def __init__(self, freqs=512, hidden_dim=64):
        super(Encoder, self).__init__()
        self.config = NetworkConfig()

        # Reducción dimensional
        hidden_1 = int(freqs / 2)
        hidden_2 = int(freqs / 4)

        # Stride 1 asegura mantener constante la dimensión temporal
        self.f_downsample_conv_1 = Conv1DBlock(freqs, hidden_1, kernel_size=5, stride=1, padding=2)
        self.f_downsample_conv_2 = Conv1DBlock(hidden_1, hidden_2, kernel_size=5, stride=1, padding=2)
        self.f_downsample_conv_3 = Conv1DBlock(hidden_2, hidden_dim, kernel_size=5, stride=1, padding=2)

        # Mantenemos constantes los canales (cada canal representa un bin de frecuencia) y usamos stride 2 para reducir el tiempo
        self.t_downsample_conv_1 = Conv1DBlock(freqs, freqs, kernel_size=5, stride=2, padding=2)
        self.t_downsample_conv_2 = Conv1DBlock(hidden_1, hidden_1, kernel_size=5, stride=2, padding=2)
        self.t_downsample_conv_3 = Conv1DBlock(hidden_2, hidden_2, kernel_size=5, stride=2, padding=2)

        # LSTM
        self.norm = nn.InstanceNorm1d(hidden_2)
        self.lstm = nn.LSTM(hidden_2, hidden_2, num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_2 * 2, hidden_2)


    def forward(self, x):

        x = x.squeeze(1)

        # Downsampling hasta que la frecuencia sea 128, primero tiempo luego frecuencia
        x_down2_t = self.t_downsample_conv_1(x)
        x_down2 = self.f_downsample_conv_1(x_down2_t)
        x_down4_t = self.t_downsample_conv_2(x_down2)
        x_down4 = self.f_downsample_conv_2(x_down4_t)

        # Aplicamos LSTM
        x_down4 = self.norm(x_down4)
        x_down4, _ = self.lstm(x_down4.permute(0, 2, 1))
        x_down4 = self.lstm_proj(x_down4)
        x_down4 = x_down4.permute(0, 2, 1)

        # Continuamos downsampleando
        x_down8_t = self.t_downsample_conv_3(x_down4)
        x_down8 = self.f_downsample_conv_3(x_down8_t)

        encoding = x_down8
        return encoding
