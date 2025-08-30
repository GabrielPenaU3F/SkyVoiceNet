from torch import nn

from source.config import NetworkConfig
from source.data_processing.normalizer import Normalizer
from source.network.decoders import Decoder
from source.network.encoders import Encoder


class SkyVoiceNet(nn.Module):

    def __init__(self, freqs=512, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.config = NetworkConfig()
        self.config.update(**kwargs)

        self.encoder_speech = Encoder(freqs)
        self.encoder_contour = Encoder(freqs)
        self.decoder = Decoder(freqs)
        self.final_activation = nn.ReLU()

    def forward(self, speech, contour):

        # Apply min-max normalization
        speech = Normalizer.minmax_normalize_batch(speech)

        # Speech and contour are encoded
        speech_embedding = self.encoder_speech(speech)
        contour_embedding = self.encoder_contour(contour)

        y_pred = self.decoder(speech_embedding, contour_embedding)
        y_pred = y_pred.unsqueeze(1)
        y_pred = self.final_activation(y_pred)
        return y_pred
