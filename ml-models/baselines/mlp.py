from os import path

import torch
from torch import nn

import settings


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.path = path.join(settings.models_path, self.get_name())
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_size, hidden_size, dtype=torch.float64))
        self.layers.append(nn.ReLU())
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size, dtype=torch.float64))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, output_size, dtype=torch.float64))

    def forward(self, x):
        return self.layers(x)

    def load_saved_model(self):
        state_dict = torch.load(path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    @staticmethod
    def get_name():
        return __class__.__name__
