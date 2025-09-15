import torch
from torch import nn
from torch.nn import functional as F

from source.config import NetworkConfig
from source.network.convolutional_blocks import ConvTranspose1DBlock
from source.network.residual_blocks import ResidualBlock
from source.network.residual_buffer import ResidualBuffer

class Decoder(nn.Module):

    def __init__(self, freqs=512):
        super(Decoder, self).__init__()
        self.config = NetworkConfig()
        self.residual_buffer = ResidualBuffer()

        # Dimensional reconstruction
        hidden_1 = int(freqs / 2)
        hidden_2 = int(freqs / 4)

        self.f_upsample_conv_3 = ConvTranspose1DBlock(hidden_2, hidden_2, kernel_size=5, stride=1, padding=2)
        self.f_upsample_conv_2 = ConvTranspose1DBlock(hidden_2, hidden_1, kernel_size=5, stride=1, padding=2)
        self.f_upsample_conv_1 = ConvTranspose1DBlock(hidden_1, freqs, kernel_size=5, stride=1, padding=2)

        self.t_upsample_conv_3 = ConvTranspose1DBlock(hidden_2, hidden_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_upsample_conv_2 = ConvTranspose1DBlock(hidden_1, hidden_1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_upsample_conv_1 = ConvTranspose1DBlock(freqs, freqs, kernel_size=3, stride=2, padding=1, output_padding=1)


        # Recurrent layer
        self.norm = nn.InstanceNorm1d(freqs)
        self.recurrent = nn.LSTM(freqs, freqs, num_layers=1, batch_first=True, bidirectional=False)

        # Residual blocks
        self.residual_block_3 = ResidualBlock(hidden_2)
        self.residual_block_2 = ResidualBlock(hidden_1)
        self.residual_block_1 = ResidualBlock(freqs)

    def forward(self, x, contour):

        x = torch.cat((x, contour), dim=1)

        # # Upsampling with residual connections

        # Upsampling 1
        x_up4_f = self.f_upsample_conv_3(x)
        x_up4 = self.t_upsample_conv_3(x_up4_f)

        # Residual block 3
        if self.config.residuals is not None:
            res_3 = self.residual_buffer.retrieve_buffer_conv_2_output()
            res3_out = self.residual_block_3(res_3)
            x_up4 = self.apply_skip_connection(x_up4, res3_out)

        # Upsampling 2
        x_up2_f = self.f_upsample_conv_2(x_up4)
        x_up2 = self.t_upsample_conv_2(x_up2_f)

        # Residual block 2
        if self.config.residuals is not None:
            res_2 = self.residual_buffer.retrieve_buffer_conv_1_output()
            res2_out = self.residual_block_2(res_2)
            x_up2 = self.apply_skip_connection(x_up2, res2_out)

        # Upsampling 3
        x_up1_f = self.f_upsample_conv_1(x_up2)
        x_up1 = self.t_upsample_conv_1(x_up1_f)

        # Residual block 1
        if self.config.residuals is not None:
            res_1 = self.residual_buffer.retrieve_input()
            res1_out = self.residual_block_1(res_1)
            x_up1 = self.apply_skip_connection(x_up1, res1_out)

        # Apply recurrent layer
        y = self.norm(x_up1)
        y, _ = self.recurrent(y.permute(0, 2, 1))
        y = y.permute(0, 2, 1)

        return y

    def apply_skip_connection(self, x, x_res):
        if not x.shape == x_res.shape:
            x = F.interpolate(x.unsqueeze(1), size=x_res.shape[-2:], mode="bilinear", align_corners=True).squeeze()

        return x + x_res
