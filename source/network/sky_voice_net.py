import torch
from torch import nn
from torch.nn import init

from source.network.activations import *
from source.network.attention_block import AttentionBlock
from source.network.decoder_block import DecoderBlock
from source.network.encoder_block import EncoderBlock


class SkyVoiceNet(nn.Module):

    def __init__(self, **kwargs):
        super(SkyVoiceNet, self).__init__()
        self.encoder_block = EncoderBlock(**kwargs)
        self.attention_block = AttentionBlock(**kwargs)
        self.decoder_block = DecoderBlock(**kwargs)
        self.act = BoundedSwish(beta=0.1)
        # self.act = NormalizedTanh()
        self.apply(self.init_weights)

    def forward(self, speech_spec, melody_contour):

        speech_embedding = self.encoder_block(speech_spec)
        aligned_embedding = self.attention_block(speech_embedding, melody_contour, speech_spec)
        output_spectrogram = self.decoder_block(aligned_embedding)

        # Final activation
        output_spectrogram = self.act(output_spectrogram)

        return output_spectrogram

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)
