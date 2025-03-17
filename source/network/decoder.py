import torch
from torch import nn

from source.config import NetworkConfig
from source.network.convolutional_block import Conv1DBlock, ConvTranspose1DBlock


class Decoder(nn.Module):
    """ Decoder de Speech o Melodía basado en convoluciones 1D """
    def __init__(self, freqs=512):
        super(Decoder, self).__init__()
        self.config = NetworkConfig()

        # Reducción dimensional
        hidden_1 = int(freqs / 2)
        hidden_2 = int(freqs / 4)

        # Stride 1 asegura mantener constante la dimensión temporal
        self.f_upsample_conv_3 = ConvTranspose1DBlock(hidden_2, hidden_2, kernel_size=5, stride=1, padding=2)
        self.f_upsample_conv_2 = ConvTranspose1DBlock(hidden_2, hidden_1, kernel_size=5, stride=1, padding=2)
        self.f_upsample_conv_1 = ConvTranspose1DBlock(hidden_1, freqs, kernel_size=5, stride=1, padding=2)

        # Mantenemos constantes los canales (cada canal representa un bin de frecuencia) y usamos stride 2 para reducir el tiempo
        self.t_upsample_conv_3 = ConvTranspose1DBlock(hidden_2, hidden_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_upsample_conv_2 = ConvTranspose1DBlock(hidden_1, hidden_1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_upsample_conv_1 = ConvTranspose1DBlock(freqs, freqs, kernel_size=3, stride=2, padding=1, output_padding=1)

        # LSTM
        self.norm = nn.InstanceNorm1d(freqs)
        self.lstm = nn.LSTM(freqs, freqs, num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)
        self.lstm_proj = nn.Linear(freqs * 2, freqs)



    def forward(self, x):

        # Upsampling
        x_up4_f = self.f_upsample_conv_3(x)
        x_up4 = self.t_upsample_conv_3(x_up4_f)
        x_up2_f = self.f_upsample_conv_2(x_up4)
        x_up2 = self.t_upsample_conv_2(x_up2_f)
        x_up1_f = self.f_upsample_conv_1(x_up2)
        x_up1 = self.t_upsample_conv_1(x_up1_f)

        # Aplicamos LSTM
        y, _ = self.lstm(x_up1.permute(0, 2, 1))
        y = self.lstm_proj(y)
        y = y.permute(0, 2, 1)

        return y
