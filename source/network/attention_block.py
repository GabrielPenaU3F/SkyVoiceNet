from torch import nn
from torch.nn import functional as F

from source.config import NetworkConfig
from source.network.positional_encoding import PositionalEncoding


class AttentionBlock(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__()
        self.config = NetworkConfig() # This one is a singleton
        self.config.update(**kwargs)
        self.positional_encoding = PositionalEncoding(d_model=self.config.transf_embedding_dim, max_len=5000)
        self.attention = nn.MultiheadAttention(self.config.transf_embedding_dim, num_heads=self.config.cross_attention_num_heads,
                                               dropout=self.config.cross_attention_dropout, batch_first=True)

    def forward(self, speech_embedding, melody_contour):

        # Format contour and project
        melody_contour = self.format_contour(melody_contour)

        # Positional encode
        speech_embedding = self.positional_encoding(speech_embedding)
        melody_contour = self.positional_encoding(melody_contour)

        # Attend
        attention_output, attn_output_weights = self.attention(query=speech_embedding,
                                                          key=melody_contour,
                                                          value=melody_contour)

        # attn_weights = attn_output_weights.detach().cpu().numpy()  # [batch, time, time]
        # plt.imshow(attn_weights[0], cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.title("Attention Weights")
        # plt.show()

        # Residual connection
        # attn_output = attn_output + speech_embedding

        return attention_output


    def format_contour(self, melody_contour):
        melody_contour = melody_contour.to(self.config.device)
        batch_size, _, num_freqs, time = melody_contour.shape  # Will be the 128 MIDIs in this implementation
        melody_contour = melody_contour.permute(0, 1, 3, 2)  # [batch, channel, time, frequencies]
        melody_contour = F.interpolate(melody_contour, size=(time, self.config.transf_embedding_dim), mode='bilinear')
        return melody_contour.squeeze(1)
