import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
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


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1, pad_idx=None):
        super(TransformerEncoderDecoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        # pad_idx=None (default) preserves the original forward() behavior
        # exactly - no padding mask is built or passed, so pre-existing
        # checkpoints and analysis code that construct this model without
        # pad_idx keep producing bit-identical inference output. Pass an
        # explicit pad_idx (e.g. 0) to opt into src/tgt padding masks.
        self.pad_idx = pad_idx

        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        # Disable the encoder's nested-tensor fastpath: it only activates when
        # a key_padding_mask is passed (never the case for pad_idx=None, so
        # this doesn't affect old/no-mask behavior at all), and the op it
        # needs (aten::_nested_tensor_from_mask_left_aligned) isn't
        # implemented for the MPS backend - without this, any pad_idx-enabled
        # forward pass crashes immediately on MPS. Note: forward() actually
        # gates on `use_nested_tensor`, a separate attribute snapshotted from
        # `enable_nested_tensor` at construction time - setting
        # enable_nested_tensor post-construction is a no-op; use_nested_tensor
        # is the one that must be set.
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # Generate causal mask for decoder
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, return_attention=False):
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1)).to(tgt.device)

        # Padding masks: only built when pad_idx is explicitly set (see
        # __init__ comment) - shape [batch, seq_len], True = ignore that
        # position as a key. memory_key_padding_mask reuses src_key_padding_mask
        # since the encoder memory has the same padding layout as src.
        if self.pad_idx is not None:
            src_key_padding_mask = (src == self.pad_idx)
            tgt_key_padding_mask = (tgt == self.pad_idx)
        else:
            src_key_padding_mask = None
            tgt_key_padding_mask = None

        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        if not return_attention:
            # original path - untouched (fast training) when pad_idx is None;
            # with pad_idx set, padding masks flow through here identically
            # to the attention-extraction path below.
            output = self.transformer(
                src_emb, tgt_emb, tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            return self.fc_out(output)

        # attention path
        encoder_attention = []
        enc_out = src_emb

        for layer in self.transformer.encoder.layers:
            # 1. self-attention with weights
            attn_out, attn_w = layer.self_attn(
                enc_out, enc_out, enc_out,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False  # [batch, heads, src, src]
            )
            # 2. rest of encoder layer manually
            enc_out = layer.norm1(enc_out + layer.dropout1(attn_out))
            ff_out  = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(enc_out)))
            )
            enc_out = layer.norm2(enc_out + layer.dropout2(ff_out))
            encoder_attention.append(attn_w.detach().cpu())

        decoder_attention = []
        dec_out = tgt_emb

        for layer in self.transformer.decoder.layers:
            # 1. decoder self-attention (causal)
            self_out, _ = layer.self_attn(
                dec_out, dec_out, dec_out,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False
            )
            dec_out = layer.norm1(dec_out + layer.dropout1(self_out))

            # 2. cross-attention with weights
            cross_out, cross_w = layer.multihead_attn(
                dec_out, enc_out, enc_out,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False  # [batch, heads, tgt, src]
            )
            dec_out = layer.norm2(dec_out + layer.dropout2(cross_out))

            # 3. feedforward
            ff_out  = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(dec_out)))
            )
            dec_out = layer.norm3(dec_out + layer.dropout3(ff_out))

            decoder_attention.append({
                'cross_attention': cross_w.detach().cpu()
            })

        logits = self.fc_out(dec_out)
        return logits, {
            'encoder_attention': encoder_attention,
            'decoder_attention': decoder_attention
        }


def create_transformer_model(vocab_size, d_model=256, nhead=8,
                             num_encoder_layers=3, num_decoder_layers=3, pad_idx=None):
    model = TransformerEncoderDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout=0.1,
        pad_idx=pad_idx,
    )
    return model