from typing import Tuple
import torch
import torch.nn as nn

class Encoder(nn.Module):
    # LSTM Encoder for sequence data
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float = 0.5) -> None:

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size 

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)  
        
        # Create a bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True  
        )

    def forward(self, input_sqe: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    
        embedded = self.dropout(self.embedding(input_sqe))  
        outputs, (hidden, cell) = self.lstm(embedded)  

        hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)  
        cell = torch.cat((cell[0:1], cell[1:2]), dim=2)     

        return outputs, (hidden, cell)  
    
class Decoder(nn.Module):
    # LSTM Decoder for sequence data
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float = 0.5) -> None:

        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.dropout = nn.Dropout(dropout)  

        # Create a unidirectional LSTM layer
        self.lstm = nn.LSTM(                                      
            input_size=embedding_dim,
            hidden_size=hidden_size * 2,  
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  
        self.output_dropout = nn.Dropout(dropout)  
        
    def forward(self, input_sqe: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        
        embedded = self.dropout(self.embedding(input_sqe))  
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))  
        predictions = self.fc(self.output_dropout(outputs))  
        return predictions 
    
class Seq2Seq(nn.Module):
    # Sequence-to-Sequence model combining Encoder and Decoder
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(src)  
        predictions = self.decoder(trg, hidden, cell)  
        return predictions  
    
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