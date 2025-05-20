import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.5):
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Transpose for CNN
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)  # (batch_size, seq_len/4, 128)
        
        # LSTM layers
        x, _ = self.lstm(x)
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        # Dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim=12, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, num_classes=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layer to convert input to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Transpose for transformer: (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=0)
        
        # Output layer
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
