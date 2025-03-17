from torch import nn

from source.config import NetworkConfig



class AttentionBlock(nn.Module):

    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.attention = nn.MultiheadAttention(embed_dim=self.config.embed_dim, num_heads=self.config.attn_heads, batch_first=True)

    def forward(self, *args):
        pass


class SelfAttentionBlock(AttentionBlock):

    def forward(self, embedding):
        attended_embedding, _ = self.attention(query=embedding,
                                               key=embedding,
                                               value=embedding)
        return attended_embedding


class FullAttentionBlock(AttentionBlock):

    def __init__(self):
        super().__init__()
        self.speech_self_attention = nn.MultiheadAttention(embed_dim=self.config.embed_dim,
                                                           num_heads=self.config.attn_heads, batch_first=True)
        self.melody_self_attention = nn.MultiheadAttention(embed_dim=self.config.embed_dim,
                                                           num_heads=self.config.attn_heads, batch_first=True)

    def forward(self, speech_embedding, melody_embedding):
        attended_speech, _ = self.speech_self_attention(speech_embedding, speech_embedding, speech_embedding)
        attended_melody, _ = self.melody_self_attention(melody_embedding, melody_embedding, melody_embedding)
        attended_embedding, _ = self.attention(query=attended_speech,
                                               key=attended_melody,
                                               value=attended_melody)
        return attended_embedding


class CrossAttentionBlock(AttentionBlock):

    def forward(self, speech_embedding, melody_embedding):
        attended_embedding, _ = self.attention(query=speech_embedding,
                                               key=melody_embedding,
                                               value=melody_embedding)
        return attended_embedding