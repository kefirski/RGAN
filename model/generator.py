import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functional import parameters_allocation_check
from torch_modules.other.highway import Highway


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params

        self.rnn = nn.ModuleList([nn.GRU(input_size=size, hidden_size=self.params.gen_size[i + 1], batch_first=True)
                                  for i, size in enumerate(self.params.gen_size[:-1])])

        self.batch_norm = nn.ModuleList([t.nn.BatchNorm2d(1) for size in self.params.gen_size[1:]])

        self.highway = nn.ModuleList([Highway(size, 2, F.relu) for size in self.params.gen_size[1:]])

        self.fc = nn.Linear(self.params.gen_size[-1], self.params.vocab_size)

    def forward(self, z, seq_len):
        """
        :param z: An tensor with shape of [batch_size, latent_variable_size] to condition generator from 
        :param seq_len: length of generated sequence
        :return: An tensor with shape of [batch_size, seq_len, vocab_size] 
                    containing probability disctribution over various words in vocabulary
        """

        [batch_size, latent_variable_size] = z.size()
        assert latent_variable_size == self.params.latent_variable_size, 'Invalid input size'

        '''for now z is [batch_size, seq_len, variable_size] shaped tensor'''
        z = z.repeat(1, 1, seq_len).view(batch_size, seq_len, latent_variable_size)

        for i, layer in enumerate(self.rnn):
            z, _ = layer(z)
            z = z.unsqueeze(1)
            z = self.batch_norm[i](z)
            z = z.squeeze(1)
            z = self.highway[i](z)

        z = z.view(-1, self.params.gen_size[-1])
        z = F.softmax(self.fc(z))
        z = z.view(batch_size, seq_len, self.params.vocab_size)

        return z
