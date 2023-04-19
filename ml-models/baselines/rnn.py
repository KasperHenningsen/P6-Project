import os

import torch
import settings
from torch import nn


class RecurrentNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0, nonlinearity='tanh'):
        super(RecurrentNeuralNet, self).__init__()
        self.path = os.path.join(settings.models_path, self.get_name())
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = 0
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout_prob
        self.nonlinearity = nonlinearity

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout_prob, nonlinearity=nonlinearity, dtype=torch.float64)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, dtype=torch.float64)
        self.to(settings.device)

    def forward(self, x):
        self.batch_size = x.shape[0]
        h0 = self.init_internal_states()
        out, hn = self.rnn(x, h0.detach())
        out = self.linear(out)
        return out.reshape(-1, 12)

    def init_internal_states(self):
        h0 = torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(settings.device)
        return h0

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    @staticmethod
    def get_name():
        return __class__.__name__
