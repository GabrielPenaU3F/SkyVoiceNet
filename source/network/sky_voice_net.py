from torch import nn
from torch.nn import init

from source.network.attention_block import AttentionBlock
from source.network.decoder_block import DecoderBlock
from source.network.encoder_block import EncoderBlock


class SkyVoiceNet(nn.Module):

    def __init__(self, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.encoder_block = EncoderBlock(**kwargs)
        self.attention_block = AttentionBlock(**kwargs)
        self.decoder_block = DecoderBlock(**kwargs)
        self.apply(self.init_weights)

    def forward(self, speech_spec, contour_spec):

        speech_embedding = self.encoder_block(speech_spec)
        attention_output = self.attention_block(speech_embedding, contour_spec)
        output_spectrogram = self.decoder_block(attention_output, memory=speech_embedding)

        return output_spectrogram

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)
