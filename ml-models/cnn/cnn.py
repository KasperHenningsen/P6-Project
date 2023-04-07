import os

import settings
import torch
from torch import nn


class Conv1D(nn.Module):
    def __init__(self, input_channels, kernel_size, output_size=1, dropout_prob=0.2):
        super().__init__()
        self.path = os.path.join(settings.models_path, self.get_name())
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=input_channels,
                                kernel_size=kernel_size, stride=1, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(input_channels, output_size, dtype=torch.float64)
        self.to(settings.device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv1d(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = out.view(x.size(0), -1)
        out = self.linear(out)
        return out.reshape(-1, 12)

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    def get_name(self):
        return self.__class__.__name__
