from os import path

import torch
from torch import nn

import settings


class MultiLayerPerceptronNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, seq_length=12):
        super(MultiLayerPerceptronNet, self).__init__()
        self.path = path.join(settings.models_path, self.get_name())
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_length = seq_length

        self.layers = nn.Sequential().to(settings.device)
        self.layers.append(nn.Linear(input_size * seq_length, hidden_size, dtype=torch.float64))
        self.layers.append(nn.ReLU())
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size, dtype=torch.float64))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, output_size * seq_length, dtype=torch.float64))
        self.to(settings.device)

    def forward(self, x):
        out = x.reshape(-1, self.input_size*self.seq_length)
        return self.layers(out)

    def load_saved_model(self):
        state_dict = torch.load(path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    @staticmethod
    def get_name():
        return __class__.__name__
