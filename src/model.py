"""From-scratch Transformer blocks (notebook transcription)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import PAD_token


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
        out = scaled_attention(q, k, v, mask)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, self.d_model)
        return self.dropout(self.linear(out))


class Encoder_feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Encoder_feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


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
        x = self.norm1(self.dropout1(self.attention(x, mask)) + residual)
        residual = x
        x = self.norm2(self.dropout2(self.feedforward(x)) + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_feedforward, dropout) for _ in range(num_layers)]
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
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class CrossHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
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
        q = q.reshape(batch_size, tgt_len, self.heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        kv = kv.reshape(batch_size, src_len, self.heads, 2 * self.head_dim).permute(0, 2, 1, 3).contiguous()
        k, v = kv.chunk(2, dim=-1)
        out = scaled_attention(q, k, v, mask=mask)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch_size, tgt_len, self.d_model)
        return self.dropout(self.linear(out))


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
        y = self.layernorm1(self.dropout1(self.self_attn(y, mask=decoder_mask)) + _y)
        _y = y
        y = self.layernorm2(self.dropout2(self.encoder_decoder_crs_attn(y, x, mask=encoder_mask)) + _y)
        _y = y
        y = self.layernorm3(self.dropout3(self.ffn(y)) + _y)
        return y


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecodeLayer(d_model, d_feedforward, num_heads, dropout=0.1) for _ in range(num_layers)]
        )

    def forward(self, x, y, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            y = layer(x, y, tgt_mask, src_mask)
        return y


class TranslateModel(nn.Module):
    pad_idx = PAD_token

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, dff, dropout=0.1, max_seq_len=65):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_token)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_token)
        self.position_embedd = Positional_embedding(d_model, self.max_seq_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dropout)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src_emb = src_emb + self.position_embedd(src.size(1))
        tgt_emb = tgt_emb + self.position_embedd(tgt.size(1))
        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(enc_output, tgt_emb, tgt_mask, src_mask)
        return self.final_linear(dec_output)

    def create_src_mask(self, src):
        return (src != PAD_token).unsqueeze(1).unsqueeze(2)

    def create_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        device = tgt.device
        causal = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1) == 0
        padding = tgt != PAD_token
        mask = causal.unsqueeze(0) & padding.unsqueeze(1)
        return mask.unsqueeze(1)
