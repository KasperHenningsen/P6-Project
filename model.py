import torch
import math
import numpy as np


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, name):
        super(MultiLayerPerceptron, self).__init__()
        self.layer_1 = torch.nn.Linear(2, 3)
        self.layer_2 = torch.nn.Linear(3, 1)
        self.name = name

    def forward(self, x):
        out = torch.nn.functional.relu(self.layer_1(x))
        out = torch.nn.functional.linear(self.layer_2(out))
        return out

    def __str__(self):
        return f'Name: {self.name}\n' + super(MultiLayerPerceptron, self).__str__()

    @staticmethod
    def sse(x1, x2):
        return np.sum((x1 - x2) ** 2)

    @staticmethod
    def mse(x1, x2):
        sse = MultiLayerPerceptron.sse(x1, x2)
        return 1 / len(x1) * sse

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.e ** -x)

    @staticmethod
    def identity(x):
        return x
