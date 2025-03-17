import torch
from torch import nn
from torch.nn import functional as F

from source.config import NetworkConfig
from source.network.attention_blocks import SelfAttentionBlock
from source.network.convolutional_blocks import ConvTranspose1DBlock
from source.network.residual_buffer import ResidualBuffer

class Decoder(nn.Module):

    def __init__(self, freqs=512):
        super(Decoder, self).__init__()
        self.config = NetworkConfig()
        self.residual_buffer = ResidualBuffer()

        # Dimensional reconstruction
        hidden_1 = int(freqs / 2)
        hidden_2 = int(freqs / 4)

        self.f_upsample_conv_3 = None # Defined by subclasses
        self.f_upsample_conv_2 = ConvTranspose1DBlock(hidden_2, hidden_1, kernel_size=5, stride=1, padding=2)
        self.f_upsample_conv_1 = ConvTranspose1DBlock(hidden_1, freqs, kernel_size=5, stride=1, padding=2)

        self.t_upsample_conv_3 = ConvTranspose1DBlock(hidden_2, hidden_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_upsample_conv_2 = ConvTranspose1DBlock(hidden_1, hidden_1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_upsample_conv_1 = ConvTranspose1DBlock(freqs, freqs, kernel_size=3, stride=2, padding=1, output_padding=1)

        # LSTM
        self.norm = nn.InstanceNorm1d(freqs)
        self.lstm = nn.LSTM(freqs, freqs, num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)
        self.lstm_proj = nn.Linear(freqs * 2, freqs)


    def forward(self, x):

        # # Upsampling with skip connections

        # Upsampling 1
        x_up4_f = self.f_upsample_conv_3(x)
        x_up4 = self.t_upsample_conv_3(x_up4_f)

        # Upsampling 2
        x_up4 = self.apply_skip_connection(x_up4, self.residual_buffer.retrieve_buffer_conv_2_output())
        x_up2_f = self.f_upsample_conv_2(x_up4)
        x_up2 = self.t_upsample_conv_2(x_up2_f)

        # Upsampling 3
        x_up2 = self.apply_skip_connection(x_up2, self.residual_buffer.retrieve_buffer_conv_1_output())
        x_up1_f = self.f_upsample_conv_1(x_up2)
        x_up1 = self.t_upsample_conv_1(x_up1_f)

        # Apply LSTM
        y, _ = self.lstm(x_up1.permute(0, 2, 1))
        y = self.lstm_proj(y)
        y = y.permute(0, 2, 1)

        return y


class ShortDecoder(nn.Module):

    # This decoder is meant to be used with an embedding dim of 64, ideally after a cross attention layer
    def __init__(self, freqs=512):
        super(ShortDecoder, self).__init__()

        hidden_2 = int(freqs / 4)
        self.f_upsample_conv_3 = ConvTranspose1DBlock(self.config.embed_dim, hidden_2, kernel_size=5, stride=1, padding=2)


    def forward(self, x):
        x = self.apply_skip_connection(x, self.residual_buffer.retrieve_buffer_conv_3_output())
        y = super().forward(x)
        return y

    def apply_skip_connection(self, x, x_res):
        if not x.shape == x_res.shape:
            x = F.interpolate(x.unsqueeze(1), size=x_res.shape[-2:], mode="bilinear", align_corners=True).squeeze()

        return x + x_res


class LargeDecoder(Decoder):

    # This decoder is meant to be used with an embedding dimension of 128, after a concatenation
    def __init__(self, freqs=512):
        super().__init__(freqs)
        hidden_2 = int(freqs / 4)
        self.f_upsample_conv_3 = ConvTranspose1DBlock(hidden_2, hidden_2, kernel_size=5, stride=1, padding=2)
        self.attention = SelfAttentionBlock()

    def forward(self, speech_embedding, melody_embedding):

        speech_embedding = self.apply_skip_connection(speech_embedding, self.residual_buffer.retrieve_buffer_conv_3_output())
        x = torch.cat((speech_embedding, melody_embedding), dim=1)

        # Optional self-attention
        x = self.attention(query=x,
                           key=x,
                           value=x)

        y = super().forward(x)

        return y
