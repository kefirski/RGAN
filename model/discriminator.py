import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import parameters_allocation_check
from torch_modules.other.highway import Highway


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params

        self.rnn = nn.ModuleList([nn.GRU(input_size=size, hidden_size=self.params.dis_size[i + 1], batch_first=True)
                                  for i, size in enumerate(self.params.dis_size[:-1])])

        self.highway = nn.ModuleList([Highway(size, 2, F.leaky_relu) for size in self.params.dis_size[1:]])

        self.fc = nn.Linear(self.params.dis_size[-1], 1)

    def forward(self):
        pass
