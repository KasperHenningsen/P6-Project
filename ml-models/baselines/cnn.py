import os

import settings
import torch
from torch import nn


class Convolution1D(nn.Module):
    def __init__(self, input_channels, hidden_size, kernel_size, dropout_prob=0.2):
        super(Convolution1D, self).__init__()
        self.path = os.path.join(settings.models_path, self.get_name())
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob

        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=hidden_size * kernel_size,
                                kernel_size=kernel_size, stride=1, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(hidden_size * kernel_size, hidden_size, dtype=torch.float64)
        self.to(settings.device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv1d(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = out.view(x.shape[0], -1)
        out = self.linear(out)
        return out

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    @staticmethod
    def get_name():
        return __class__.__name__
