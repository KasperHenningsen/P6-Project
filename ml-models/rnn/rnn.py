import torch
from torch import nn


class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0, nonlinearity='tanh'):
        super(RNNNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout_prob, nonlinearity=nonlinearity, dtype=torch.float64)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, dtype=torch.float64)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(1*self.num_layers, batch_size, self.hidden_size, dtype=torch.float64)
        out, hn = self.rnn(x, h0.detach())
        out = self.linear(out)
        return out.reshape(-1, 12)
