from torch import nn

from source.config import NetworkConfig
from source.network.cross_attention_layer import CrossAttentionLayer


class AttentionBlock(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.contour_projection_layer = None
        self.attention_layer = None

    def forward(self, speech_embedding, melody_contour):

        melody_contour = self.format_attention_input(speech_embedding, melody_contour)

        if self.attention_layer is None or self.contour_projection_layer is None:
            embedding_dim = speech_embedding.shape[-1]
            self.contour_projection_layer = nn.Linear(melody_contour.shape[-1], embedding_dim).to(melody_contour.device)
            self.attention_layer = CrossAttentionLayer(embedding_dim, self.config.cross_attention_num_heads,
                                                       self.config.cross_attention_dropout, device=self.config.device)

        melody_contour = self.contour_projection_layer(melody_contour)
        attention_output = self.attention_layer(speech_embedding, melody_contour)
        return attention_output

    def format_attention_input(self, speech_embedding, contour_spec):
        _, speech_time, _ = speech_embedding.shape
        batch_size, _, num_freqs, _ = contour_spec.shape  # Will be the 128 MIDIs in this implementation
        melody_contour = nn.functional.interpolate(contour_spec,
                                                   size=(num_freqs, speech_time), mode="bilinear", align_corners=False)
        melody_contour = melody_contour.view(batch_size, num_freqs, -1).permute(0, 2, 1)  # [batch, time, frequencies]
        return melody_contour
