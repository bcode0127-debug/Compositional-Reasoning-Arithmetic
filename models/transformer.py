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
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

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
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # Generate causal mask for decoder
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, return_attention=False):
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1)).to(tgt.device)

        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        if not return_attention:
            # original path -untouched (fast training)
            output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.fc_out(output)

        # attention path 
        encoder_attention = []
        enc_out = src_emb

        for layer in self.transformer.encoder.layers:
            # 1. self-attention with weights
            attn_out, attn_w = layer.self_attn(
                enc_out, enc_out, enc_out,
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
                need_weights=False
            )
            dec_out = layer.norm1(dec_out + layer.dropout1(self_out))

            # 2. cross-attention with weights
            cross_out, cross_w = layer.multihead_attn(
                dec_out, enc_out, enc_out,
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
                             num_encoder_layers=3, num_decoder_layers=3):
    model = TransformerEncoderDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout=0.1  
    )
    return model