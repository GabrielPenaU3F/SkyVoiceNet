from torch import nn

from source.network.attention_block import AttentionBlock
from source.network.speech_encoder_block import SpeechEncoderBlock


class SkyVoiceNet(nn.Module):

    def __init__(self, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.speech_encoder_block = SpeechEncoderBlock(**kwargs)
        self.attention_block = AttentionBlock(**kwargs)
        # self.decoder = Decoder()  # Decodifica la salida final

    def forward(self, speech_spec, contour_spec):

        speech_embedding = self.speech_encoder_block(speech_spec)

        attention_output = self.attention_block(speech_embedding, contour_spec)

        print(attention_output.shape)
        a = 2

        # Decodificar la salida final
        # output = self.decoder(attended_features)
        # return output
