import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import parameters_allocation_check
from torch_modules.other.highway import Highway


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params
        self.rnn = nn.ModuleList([nn.GRU(input_size=self.params.latent_variable_size,
                                         hidden_size=self.params.decoder_size[0],
                                         batch_first=True),
                                  nn.GRU(input_size=self.params.decoder_size[0],
                                         hidden_size=self.params.decoder_size[1],
                                         batch_first=True),
                                  nn.GRU(input_size=self.params.decoder_size[1],
                                         hidden_size=self.params.decoder_size[2],
                                         batch_first=True)])

        self.batch_norm = nn.ModuleList([torch.nn.BatchNorm2d(self.params.decoder_size[0]),
                                         torch.nn.BatchNorm2d(self.params.decoder_size[1]),
                                         torch.nn.BatchNorm2d(self.params.decoder_size[2])])

        self.highway = nn.ModuleList([Highway(self.params.decoder_size[0], 2, F.prelu),
                                      Highway(self.params.decoder_size[1], 2, F.prelu),
                                      Highway(self.params.decoder_size[2], 3, F.prelu)
                                      ])

        self.fc = nn.Linear(self.params.decoder_size[2], self.params.vocab_size)

    def forward(self, z, seq_len):
        assert z.size()[1] == self.params.latent_variable_size, 'Invalid input size'
        [batch_size, _] = z.size()

        # for now z is [batch_size, seq_len, variable_size] shaped tensor
        z = z.repeat.repeat(1, 1, seq_len).view(batch_size, seq_len, self.params.latent_variable_size)

        for i, layer in enumerate(self.rnn):
            z, _ = layer(z)
            z = self.batch_norm[i](z)
            z = self.highway[i](z)

        z = z.view(-1, self.params.decoder_size[2])
        z = F.softmax(self.fc(z))
        z = z.view(batch_size, seq_len, self.params.vocab_size)

        return z
