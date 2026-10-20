"""Positional-encoding schemes for the positional-encoding sweep.

Design: docs/design/pos-encoding-sweep.md (pre-registration 9365283). The axis
is where position enters the model:

- once, at the input: ``sinusoidal`` (the paper's scheme, unchanged) and
  ``learned`` (a learned absolute table);
- re-injected in every self-attention layer: ``rope`` (rotation of q and k)
  and ``alibi`` (a per-head distance bias on the attention scores);
- ``nope``: no position signal in the encoder (floor check only).

The re-injecting schemes live in ``PositionalMultiheadAttention``, a subclass
of ``nn.MultiheadAttention`` with the same parameter names, and
``StandardPathEncoderLayer``, which keeps ``nn.TransformerEncoderLayer``'s
parameters but never takes PyTorch's fused fast path (that path reads
``in_proj_weight`` directly and would skip the rotation or bias).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

POSITIONAL_ENCODINGS = ("sinusoidal", "learned", "rope", "alibi", "nope")
LEARNED_MAX_LEN = 64  # the paper's input budget (MAX_INPUT_LEN)
ROPE_BASE = 10000.0


class PositionalEncoding(nn.Module):
    """The paper's sinusoidal encoding (moved here unchanged from models/transformer.py)."""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding for batch_first=True
        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        # Add positional encoding
        return x + self.pe[:, :x.size(1), :]


def sinusoidal_rms(d_model, max_len=LEARNED_MAX_LEN):
    """RMS of the sinusoidal table over positions 0..max_len-1 (sqrt(1/2) for even d_model)."""
    return float(PositionalEncoding(d_model, max_len=max_len).pe.pow(2).mean().sqrt())


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positions, added to the input like the sinusoidal table."""

    def __init__(self, d_model, max_len=LEARNED_MAX_LEN, std=None, generator=None):
        super().__init__()
        self.std = sinusoidal_rms(d_model, max_len) if std is None else float(std)
        self.weight = nn.Parameter(torch.empty(max_len, d_model))
        with torch.no_grad():
            self.weight.normal_(0.0, self.std, generator=generator)

    def forward(self, x):
        if x.size(1) > self.weight.size(0):
            raise ValueError(f"sequence length {x.size(1)} exceeds the learned table ({self.weight.size(0)})")
        return x + self.weight[: x.size(1)].unsqueeze(0)


class NoPositionalEncoding(nn.Module):
    """Identity: no position signal (kept as a module so it can carry a hook)."""

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# RoPE (Su et al.): rotate consecutive pairs (x_2i, x_2i+1) by pos * base^(-2i/d)
# ---------------------------------------------------------------------------

def rope_angles(seq_len, head_dim, base=ROPE_BASE, device=None, dtype=torch.float32):
    if head_dim % 2:
        raise ValueError("RoPE needs an even head dimension")
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv_freq[None, :]  # [L, head_dim/2]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: [..., L, head_dim]; cos/sin: [L, head_dim/2]. Rotates each consecutive pair."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ---------------------------------------------------------------------------
# ALiBi (Press et al.): slopes 2^(-8h/H); symmetric |i - j| in the encoder
# (MosaicBERT), causal (i - j) for j <= i in the decoder
# ---------------------------------------------------------------------------

def alibi_slopes(num_heads):
    return torch.tensor([2.0 ** (-8.0 * h / num_heads) for h in range(1, num_heads + 1)])


def alibi_bias(q_len, k_len, num_heads, causal, device=None, dtype=torch.float32):
    """[H, q_len, k_len]: -m_h |i - j| (symmetric), or -m_h (i - j) for j <= i and -inf above (causal)."""
    i = torch.arange(q_len, device=device)[:, None]
    j = torch.arange(k_len, device=device)[None, :]
    slopes = alibi_slopes(num_heads).to(device=device, dtype=dtype)[:, None, None]
    if causal:
        bias = -slopes * (i - j).to(dtype)
        bias = bias.masked_fill((j > i)[None], float("-inf"))
    else:
        bias = -slopes * (i - j).abs().to(dtype)
    return bias


def _as_additive(mask, dtype):
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(mask, float("-inf"))
    return mask.to(dtype)


class PositionalMultiheadAttention(nn.MultiheadAttention):
    """nn.MultiheadAttention with RoPE or ALiBi applied inside attention.

    Same parameters (in_proj_weight, in_proj_bias, out_proj) as the stock module, so
    a re-injecting arm's state dict differs from the paper's only by the absent
    sinusoidal buffers. Only the configuration the paper uses is supported:
    batch_first, shared q/k/v dimension, input projection with bias.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True, mode="rope", causal=False, **kw):
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=batch_first, **kw)
        if mode not in ("rope", "alibi"):
            raise ValueError(f"mode must be rope or alibi, got {mode!r}")
        if not (self.batch_first and self._qkv_same_embed_dim and self.in_proj_bias is not None):
            raise ValueError("PositionalMultiheadAttention supports batch_first, shared-dim, biased projections only")
        self.mode = mode
        self.causal = causal

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        B, L, E = query.shape
        S = key.size(1)
        H, Dh = self.num_heads, self.head_dim
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q = F.linear(query, w_q, b_q).view(B, L, H, Dh).transpose(1, 2)
        k = F.linear(key, w_k, b_k).view(B, S, H, Dh).transpose(1, 2)
        v = F.linear(value, w_v, b_v).view(B, S, H, Dh).transpose(1, 2)

        bias = torch.zeros(1, 1, L, S, dtype=q.dtype, device=q.device)
        if self.mode == "rope":
            cos_q, sin_q = rope_angles(L, Dh, device=q.device, dtype=q.dtype)
            cos_k, sin_k = (cos_q, sin_q) if S == L else rope_angles(S, Dh, device=q.device, dtype=q.dtype)
            q, k = apply_rope(q, cos_q, sin_q), apply_rope(k, cos_k, sin_k)
        else:
            bias = bias + alibi_bias(L, S, H, causal=self.causal, device=q.device, dtype=q.dtype)[None]
        am = _as_additive(attn_mask, q.dtype)
        if am is not None:
            bias = bias + (am.view(B, H, L, S) if am.dim() == 3 else am)
        elif is_causal:
            bias = bias + _as_additive(torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1), q.dtype)
        kpm = _as_additive(key_padding_mask, q.dtype)
        if kpm is not None:
            bias = bias + kpm[:, None, None, :]

        drop = self.dropout if self.training else 0.0
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh) + bias
            weights = scores.softmax(dim=-1)
            out = F.dropout(weights, p=drop) @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(B, H, L, S), dropout_p=drop)
        out = self.out_proj(out.transpose(1, 2).reshape(B, L, E))
        if weights is not None and average_attn_weights:
            weights = weights.mean(dim=1)
        return out, weights


class StandardPathEncoderLayer(nn.TransformerEncoderLayer):
    """nn.TransformerEncoderLayer that always takes the standard (non-fused) path."""

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
