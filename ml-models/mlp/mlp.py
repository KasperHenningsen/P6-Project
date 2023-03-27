from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        ).double()

    def forward(self, x):
        return self.layers(x)

