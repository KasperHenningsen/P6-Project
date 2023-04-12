import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dtype=torch.float64)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, dtype=torch.float64),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model, dtype=torch.float64)
        self.norm2 = nn.LayerNorm(d_model, dtype=torch.float64)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout2(src2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.path = os.path.join(settings.models_path, self.get_name())
        self.embedding = nn.Linear(input_size, d_model, dtype=torch.float64)
        self.transformer = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, output_size, dtype=torch.float64)
        self.to(settings.device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    def get_name(self):
        return self.__class__.__name__
