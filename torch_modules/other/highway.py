import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size)] * num_layers)
        self.linear = nn.ModuleList([nn.Linear(size, size)] * num_layers)
        self.gate = nn.ModuleList([nn.Linear(size, size)] * num_layers)

        self.f = f

    def forward(self, x):
        """
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """
        size = x.size()
        if len(size) == 3:
            x = x.view(-1, size[2])

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        if len(size) == 3:
            x = x.view(size[0], size[1], size[2])

        return x