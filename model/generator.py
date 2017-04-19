import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import parameters_allocation_check
from torch_modules.other.highway import Highway


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params

        self.rnn = nn.Sequential(
            nn.GRU(input_size=self.params.latent_variable_size,
                   hidden_size=self.params.decoder_size[0],
                   batch_first=True),

            torch.nn.BatchNorm2d(self.params.decoder_size[0]),

            nn.GRU(input_size=self.params.decoder_size[0],
                   hidden_size=self.params.decoder_size[1],
                   batch_first=True),

            torch.nn.BatchNorm2d(self.params.decoder_size[1]),

            nn.GRU(input_size=self.params.decoder_size[1],
                   hidden_size=self.params.decoder_size[2],
                   batch_first=True),

            torch.nn.BatchNorm2d(self.params.decoder_size[2])

        )

        self.hw = Highway(self.params.decoder_size[-1], 3, F.prelu)
        self.fc = nn.Linear(self.params.decoder_size[-1], self.params.vocab_size)

    def forward(self):
        pass
