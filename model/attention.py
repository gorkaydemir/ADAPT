import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, p_drop):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.last_projection = nn.Linear(self.all_head_size, hidden_size)
        self.attention_drop = nn.Dropout(p_drop)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_value_states, attention_mask):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = F.linear(key_value_states, self.key.weight)
        mixed_value_layer = self.value(key_value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(
            query_layer/math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_drop(attention_probs)

        assert torch.isnan(attention_probs).sum() == 0

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        # context_layer.shape = (batch, max_vector_num, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.last_projection(context_layer)
        return context_layer
