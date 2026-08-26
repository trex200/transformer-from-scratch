"""From-scratch Transformer blocks (notebook transcription)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_attention(q, k, v, mask=None):
    dk = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-1, -2))
    scaled = scaled / math.sqrt(dk)
    if mask is not None:
        scaled = scaled.masked_fill(mask == 0, float("-inf"))
    attention = F.softmax(scaled, dim=-1)
    output = torch.matmul(attention, v)
    return output


class Positional_embedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, self.heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        scaled_attention_output = scaled_attention(q, k, v, mask)
        scaled_attention_output = scaled_attention_output.permute(0, 2, 1, 3).contiguous()
        scaled_attention_output = scaled_attention_output.reshape(batch_size, seq_len, self.d_model)
        scaled_attention_output = self.linear(scaled_attention_output)
        outputs = self.dropout(scaled_attention_output)
        return outputs


class Encoder_feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Encoder_feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.feedforward = Encoder_feedforward(d_model, d_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderFeedForward(nn.Module):
    def __init__(self, d_model, ffn_layer, dropout=0.1):
        super(DecoderFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_layer)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(ffn_layer, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CrossHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        batch_size, tgt_len, _ = x.shape
        _, src_len, _ = y.shape
        q = self.q_layer(x)
        kv = self.kv_layer(y)
        q = q.reshape(batch_size, tgt_len, self.heads, self.head_dim)
        kv = kv.reshape(batch_size, src_len, self.heads, 2 * self.head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        kv = kv.permute(0, 2, 1, 3).contiguous()
        k, v = kv.chunk(2, dim=-1)
        scaled_attentions = scaled_attention(q, k, v, mask=mask)
        scaled_attentions = scaled_attentions.permute(0, 2, 1, 3).contiguous()
        scaled_attentions = scaled_attentions.reshape(batch_size, tgt_len, self.d_model)
        outputs = self.linear(scaled_attentions)
        outputs = self.dropout(outputs)
        return outputs


class DecodeLayer(nn.Module):
    def __init__(self, d_model, ffn_layer, num_heads, dropout=0.1):
        super(DecodeLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder_decoder_crs_attn = CrossHeadAttention(d_model, num_heads)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = Encoder_feedforward(d_model, ffn_layer, dropout)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, y, decoder_mask=None, encoder_mask=None):
        _y = y
        y = self.self_attn(y, mask=decoder_mask)
        y = self.dropout1(y)
        y = self.layernorm1(y + _y)
        _y = y
        y = self.encoder_decoder_crs_attn(y, x, mask=encoder_mask)
        y = self.dropout2(y)
        y = self.layernorm2(y + _y)
        _y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layernorm3(y + _y)
        return y


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecodeLayer(d_model, d_feedforward, num_heads, dropout=0.1)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            y = layer(x, y, tgt_mask, src_mask)
        return y
