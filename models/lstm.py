from typing import Tuple
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """LSTM Encoder for sequence data"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float = 0.5) -> None:

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size 

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)  # create dropout layer
        
        # Create a bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True  # Key feature: bidirectionality
        )

    def forward(self, input_sqe: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    
        embedded = self.dropout(self.embedding(input_sqe))  # convert input to embeddings
        outputs, (hidden, cell) = self.lstm(embedded)  # pass embeddings through LSTM

        hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)  # concatenate the final forward and backward hidden states
        cell = torch.cat((cell[0:1], cell[1:2]), dim=2)      # concatenate the final forward and backward cell states

        return outputs, (hidden, cell)  # return outputs and hidden states
    
class Decoder(nn.Module):
    """LSTM Decoder for sequence data"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float = 0.5) -> None:

        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.dropout = nn.Dropout(dropout)  # create dropout layer

        # Create a unidirectional LSTM layer
        self.lstm = nn.LSTM(                                      
            input_size=embedding_dim,
            hidden_size=hidden_size * 2,  # because encoder is bidirectional
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  # output layer to map to vocab size
        self.output_dropout = nn.Dropout(dropout)  # dropout layer for outputs
        
    def forward(self, input_sqe: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        
        embedded = self.dropout(self.embedding(input_sqe))  # convert input to embeddings
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # pass embeddings through LSTM
        predictions = self.fc(self.output_dropout(outputs))  # map LSTM outputs to vocabulary space
        return predictions # return predictions and hidden states
    
class Seq2Seq(nn.Module):
    """Sequence-to-Sequence model combining Encoder and Decoder"""
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(src)  # encode the source sequence
        predictions = self.decoder(trg, hidden, cell)  # decode the target sequence using encoder's hidden states
        return predictions  # return the final output predictions
    
def create_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_size: int = 256,
    dropout: float = 0.5
) -> Seq2Seq:
    
    # Instantiate encoder and decoder
    encoder = Encoder(vocab_size, embedding_dim, hidden_size, dropout)
    decoder = Decoder(vocab_size, embedding_dim, hidden_size, dropout)
    model = Seq2Seq(encoder, decoder)
    return model