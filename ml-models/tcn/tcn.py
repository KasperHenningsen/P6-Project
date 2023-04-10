import os.path

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import settings


class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12, depth=4, kernel_size=3, dilation_base=2):
        super(TemporalConvolutionNetwork, self).__init__()
        self.path = os.path.join(settings.models_path, self.get_name())

        self.residual_blocks = nn.Sequential(
            ResidualBlock(input_size, hidden_size, dilation=1, kernel_size=kernel_size)
        )
        for i in range(1, depth):
            dilation = dilation_base ** i
            self.residual_blocks.append(ResidualBlock(hidden_size, hidden_size, dilation, kernel_size))
        self.linear = nn.Linear(hidden_size, output_size, dtype=torch.float64)
        self.to(settings.device)

    def forward(self, x):
        out = self.residual_blocks(x)
        out = self.linear(out)
        return out.reshape(-1, 12)

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    def get_name(self):
        return self.__class__.__name__


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.padding = (kernel_size-1) * dilation
        self.pad = nn.ConstantPad1d((self.padding, 0), 0)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation,
                                           dtype=torch.float64))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation,
                                           dtype=torch.float64))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residualConv = nn.Conv1d(in_channels, out_channels, kernel_size=1, dtype=torch.float64)
        self.init_weights()
        self.to(settings.device)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.residualConv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.swapdims(1, 2)
        residual = self.residualConv(x)
        out = x
        for conv in [self.conv1, self.conv2]:
            out = self.pad(out)
            out = conv(out)
            out = self.relu(out)
            out = self.dropout(out)

        out += residual
        return self.relu(out)