import torch
from torch import nn


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout_prob=0.2):
        super(GRUNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout_prob, batch_first=True, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, dtype=torch.float64)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(1*self.num_layers, batch_size, self.hidden_size, dtype=torch.float64)
        out, hn = self.gru(x, h0.detach())
        out = self.relu(out)
        out = self.linear(out)
        return out.reshape(-1, 12)
