import os

import settings
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout_prob=0.2,
                 num_layers=2):
        super().__init__()
        self.path = os.path.join(settings.models_path, self.get_name())
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = 0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout_prob, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size, dtype=torch.float64)
        self.to(settings.device)

    def forward(self, x):
        self.batch_size = x.shape[0]
        h0, c0 = self.init_internal_states()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.relu(out)
        out = self.linear(out)
        return out.reshape(-1, 12)

    def init_internal_states(self):
        h0 = torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(settings.device)
        c0 = torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(settings.device)
        return h0, c0

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    def get_name(self):
        return self.__class__.__name__
