import torch
import torch.nn as nn
from torch.nn import Parameter


class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.a2 = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, 1)
        sigma = torch.std(z, 1)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a2.expand_as(ln_out) + self.b2.expand_as(ln_out)

        return ln_out
