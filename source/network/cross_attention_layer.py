import torch.nn as nn
from matplotlib import pyplot as plt


class CrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, device):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True, device=device)

    def forward(self, speech_embedding, melody_contour):
        """
        Here, our speech embedding is the transformer output. This layer will align it with the melody feature. Thus:

        speech_embedding: Dimensions [time, batch_size, embed_dim]
        melody_contour: Dimensions [time, batch_size, embed_dim]
        """
        query = melody_contour.permute(1, 0, 2) # We use melody features as the query
        key = speech_embedding.permute(1, 0, 2) # Speech embedding as key
        value = speech_embedding.permute(1, 0, 2) # And also as the value

        # Normalize key and query to match scaling
        query = (query - query.mean()) / (query.std() + 1e-6)
        key = (key - key.mean()) / (key.std() + 1e-6)

        attn_output, attn_output_weights = self.attention(query=query,
                                                          key=key,
                                                          value=value)
        attn_output = attn_output.permute(1, 0, 2)

        # attn_weights = attn_output_weights.detach().cpu().numpy()  # [batch, time, time]
        # plt.imshow(attn_weights[0], cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.title("Attention Weights")
        # plt.show()

        return attn_output
