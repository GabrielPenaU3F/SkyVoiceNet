import torch
from torch import nn

from source.config import NetworkConfig
from source.network.cross_attention_layer import CrossAttentionLayer
from source.network.dynamic_module import DynamicModule


class AttentionBlock(DynamicModule):

    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.fc = None

    def forward(self, speech_embedding, melody_contour):

        melody_contour = self.format_attention_input(speech_embedding, melody_contour)
        embedding_dim = speech_embedding.shape[-1]

        # Dynamically define the projection and the attention layers, only once
        contour_projection_layer = self.define_layer_dynamically("contour_projection_layer", torch.nn.Linear,
                                                                    melody_contour.shape[-1], embedding_dim, device=self.config.device)
        self.fc = nn.Sequential(
            contour_projection_layer,
            nn.LayerNorm(embedding_dim, device=self.config.device),
            nn.ReLU()
        )

        cross_attention_layer = self.define_layer_dynamically("cross_attention_layer", CrossAttentionLayer,
                                                                    embedding_dim, self.config.cross_attention_num_heads,
                                                                    self.config.cross_attention_dropout, device=self.config.device)

        # Project to match dimensions, then apply the attention
        melody_contour = self.fc(melody_contour)
        attention_output = cross_attention_layer(speech_embedding, melody_contour)
        return attention_output

    def format_attention_input(self, speech_embedding, contour_spec):
        _, speech_time, _ = speech_embedding.shape
        batch_size, _, num_freqs, _ = contour_spec.shape  # Will be the 128 MIDIs in this implementation
        melody_contour = nn.functional.interpolate(contour_spec,
                                                   size=(num_freqs, speech_time), mode="bilinear", align_corners=False)
        melody_contour = melody_contour.view(batch_size, num_freqs, -1).permute(0, 2, 1)  # [batch, time, frequencies]
        return melody_contour
