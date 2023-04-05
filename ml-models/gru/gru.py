import torch
from torch import nn


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout_prob=0.2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout_prob, batch_first=True, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, dtype=torch.float64)

    def forward(self, x):
        out, h = self.gru(x)
        out = self.relu(out)
        out = self.linear(out)
        return out.reshape(-1, 12)
