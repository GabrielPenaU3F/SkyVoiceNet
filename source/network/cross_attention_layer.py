import torch.nn as nn


class CrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, device):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True, device=device)

    def forward(self, speech_embedding, melody_contour):
        """
        Here, our speech embedding is the transformer output. This layer will align it with the melody feature. Thus:

        speech_embedding: Dimensions [batch_size, time, embed_dim]
        melody_contour: Dimensions [batch_size, time, embed_dim]
        """
        attn_output, _ = self.attention(query=melody_contour,  # We use melody features as the query
                                        key=speech_embedding, # Speech embedding as key
                                        value=speech_embedding, # And also as the value
                                        need_weights=False)
        return attn_output