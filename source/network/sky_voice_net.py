from torch import nn

from source.config import NetworkConfig
from source.network.attention_blocks import CrossAttentionBlock, FullAttentionBlock, SelfAttentionBlock
from source.network.decoders import ShortDecoder, LargeDecoder
from source.network.encoders import Encoder


class SkyVoiceNet(nn.Module):

    def __init__(self, freqs=512, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.config = NetworkConfig()
        self.config.update(**kwargs)
        self.encoder_speech = Encoder(freqs)
        self.encoder_contour = Encoder(freqs)
        self.self_attention = SelfAttentionBlock()
        self.cross_attention = CrossAttentionBlock()
        self.full_attention = FullAttentionBlock()
        self.short_decoder = ShortDecoder(freqs) # Input dim 64
        self.large_decoder = LargeDecoder(freqs) # Input dim 128
        self.final_activation = nn.LeakyReLU()

    def forward(self, speech, contour):

        # Speech and contour are encoded
        speech_embedding = self.encoder_speech(speech)
        contour_embedding = self.encoder_contour(contour)

        # # For versions with the 64-decoder

        # speech_embedding = speech_embedding.permute(0, 2, 1)
        # contour_embedding = contour_embedding.permute(0, 2, 1)

        # Cross attention
        # embedding = self.cross_attention(speech_embedding, contour_embedding)

        # Full attention
        # embedding = self.full_attention(speech_embedding, contour_embedding)

        # embedding = embedding.permute(0, 2, 1)
        # y_pred = self.short_decoder(embedding)

        # # For versions with the 128-decoder
        y_pred = self.large_decoder(speech_embedding, contour_embedding)

        y_pred = self.final_activation(y_pred.unsqueeze(1))
        return y_pred
