import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns


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


class ScaledDotProductAttention(nn.Module):

    def __init__(self, embed_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Optional masking

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # Softmax
        attn_weights = self.dropout(attn_weights)

        attn_weights_img = attn_weights.detach().cpu().numpy()  # [batch, time, time]
        plt.imshow(attn_weights_img[0], cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title("Attention Weights")
        plt.show()

        output = torch.matmul(attn_weights, value)

        return output, attn_weights
