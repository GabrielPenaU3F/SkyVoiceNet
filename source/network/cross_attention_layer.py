import math

import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt



class CrossAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # self.attention = ScaledDotProductAttention(embed_dim)

    def forward(self, speech_embedding, melody_contour):

        attn_output, attn_output_weights = self.attention(query=melody_contour, # We use melody features as the query
                                                          key=speech_embedding, # Speech embedding as key
                                                          value=speech_embedding) # And also as the value


        # sns.heatmap(attn_output_weights.cpu().numpy(), cmap="viridis")
        attn_weights = attn_output_weights.detach().cpu().numpy()  # [batch, time, time]
        plt.imshow(attn_weights[0], cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title("Attention Weights")
        plt.show()

        # Residual connection
        attn_output = attn_output + speech_embedding
        return attn_output


import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, features, query_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(features, embed_dim)
        self.value_proj = nn.Linear(features, embed_dim)

    def forward(self, speech, melody):
        Q = self.query_proj(melody)  # [batch, time, embed_dim]
        K = self.key_proj(speech)    # [batch, time_r, embed_dim]
        V = self.value_proj(speech)  # [batch, time_r, embed_dim]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)  # Se encarga del mismatch de dimensiones

        output = torch.matmul(attn_weights, V)
        return output, attn_weights
