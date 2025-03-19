from torch import nn

from source.config import NetworkConfig
from source.network.decoders import ShortDecoder, LargeDecoder
from source.network.encoders import Encoder


class SkyVoiceNet(nn.Module):

    def __init__(self, freqs=512, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.config = NetworkConfig()
        self.config.update(**kwargs)

        self.encoder_speech = Encoder(freqs)
        self.encoder_contour = Encoder(freqs)
        self.decoder = self.select_decoder(freqs)
        self.final_activation = nn.ReLU()

    def forward(self, speech, contour):

        # Speech and contour are encoded
        speech_embedding = self.encoder_speech(speech)
        contour_embedding = self.encoder_contour(contour)

        y_pred = self.decoder(speech_embedding, contour_embedding)

        y_pred = self.final_activation(y_pred.unsqueeze(1))
        return y_pred

    def select_decoder(self, freqs):

        # Decoder with 128-embeddings
        if self.config.mode in {None, 'self_attn'}:
            return LargeDecoder(freqs)

        # Decoder with 64-embeddings
        elif self.config.mode in {'cross_attn', 'double_attn'}:
            return ShortDecoder(freqs)

