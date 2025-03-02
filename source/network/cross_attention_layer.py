import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns


class CrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super(CrossAttentionLayer, self).__init__()
        #self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.attention = ScaledDotProductAttention(embed_dim)

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

        attn_output, attn_output_weights = self.attention(Q=query,
                                                          K=key,
                                                          V=value)
        attn_output = attn_output.permute(1, 0, 2)

        # sns.heatmap(attn_output_weights.cpu().numpy(), cmap="viridis")
        attn_weights = attn_output_weights.detach().cpu().numpy()  # [batch, time, time]
        plt.imshow(attn_weights[0], cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title("Attention Weights")
        plt.show()

        return attn_output


import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, embed_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.attn_weights = None

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Optional masking

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # Softmax
        output = torch.matmul(attn_weights, V)
        self.attn_weights = attn_weights.detach()

        return output, self.attn_weights
