import torch.functional as F
import torch
import torch.nn as nn

class RobertaEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()

        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()


    def forward(self, x, attention_mask=None, output_attentions=False):
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=output_attentions,
            average_attn_weights=False,
        )

        x = x + self.dropout(attn_output)
        x = self.attn_layer_norm(x)

        ffn_output = self.fc2(
            self.dropout(self.activation(self.fc1(x)))
        )

        x = x + self.dropout(ffn_output)
        x = self.ffn_layer_norm(x)

        if output_attentions:
            return x, attn_weights

        return x
