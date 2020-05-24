import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim):
        super(Perceptron, self).__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)
