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
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        # Create causal mask
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Embed with scaling
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        # Transformer forward
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        
        # Output projection
        logits = self.fc_out(output)
        return logits


def create_transformer_model(vocab_size, d_model=256, nhead=8,
                             num_encoder_layers=3, num_decoder_layers=3):
    model = TransformerEncoderDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout=0.0  # No dropout for small datasets
    )
    return model