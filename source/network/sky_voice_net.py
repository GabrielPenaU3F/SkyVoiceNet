import torch
from torch import nn

from source.config import NetworkConfig
from source.network.decoder import Decoder
from source.network.encoder import Encoder


class SkyVoiceNet(nn.Module):

    def __init__(self, freqs=512, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.config = NetworkConfig()
        self.config.update(**kwargs)
        self.encoder_speech = Encoder(freqs)
        self.encoder_contour = Encoder(freqs)
        self.attention = nn.MultiheadAttention(embed_dim=self.config.embed_dim, num_heads=self.config.attn_heads, batch_first=True)
        self.decoder = Decoder(freqs)
        self.final_activation = nn.LeakyReLU()

    def forward(self, speech, contour):

        # Codificamos
        speech_embedding = self.encoder_speech(speech)
        contour_embedding = self.encoder_contour(contour)
        embedding = torch.cat((speech_embedding, contour_embedding), dim=1)

        # Aplicamos atención
        embedding = embedding.permute(0, 2, 1)
        embedding, _ = self.attention(embedding, embedding, embedding)
        embedding = embedding.permute(0, 2, 1)

        # Decodificamos
        y_pred = self.decoder(embedding)
        y_pred = self.final_activation(y_pred.unsqueeze(1))
        return y_pred
