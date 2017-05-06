import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_modules.softargmax.softargmax import SoftArgmax
from torch_modules.layerNormGRUCell.layerNormGRUCell import LayerNormGRUCell


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params

        self.rnn = nn.ModuleList([LayerNormGRUCell(input_size=size, hidden_size=self.params.gen_size[i + 1])
                                  for i, size in enumerate(self.params.gen_size[:-1])])

        self.latent_to_hidden = nn.ModuleList([nn.Linear(self.params.latent_variable_size, size)
                                               for size in self.params.gen_size[1:]])

        self.hidden_to_vocab_size = nn.Linear(self.params.gen_size[-1], self.params.vocab_size)

        self.soft_argmax = SoftArgmax(temperature=1e-3)

    def forward(self, z, seq_len, embedding_lockup):
        """
        :param z: An tensor with shape of [batch_size, latent_variable_size] to condition generation from 
        :param seq_len: length of generated sequence
        :param embedding_lockup: An function to lockup weighted embeddings
        :return: An tensor with shape of [batch_size, seq_len, word_embed_size] 
                    containing continious generated data
        """

        [batch_size, _] = z.size()

        '''y is input to rnn at avery time step'''
        y = Variable(t.zeros(batch_size, self.params.word_embed_size))
        if z.is_cuda:
            y = y.cuda()
        y = t.cat([y, z], 1)

        '''
        hidden_state is an array of initial states 
        each with shape of [batch_size, hidden_size_i]
        '''
        hidden_state = self.initial_hidden_state(z)
        result = []

        for i in range(seq_len):
            y, hidden_state = self.unroll_cell(y, hidden_state)
            y = self.soft_argmax(y)
            y = embedding_lockup(y)
            result += [y.unsqueeze(1)]
            y = t.cat([y, z], 1)

        return t.cat(result, 1)

    def sample(self, z, seq_len, batch_loader, embedding_lockup):
        """
        :param seq_len: length of generated sequence
        :param batch_loader: BatchLoader instance
        :param embedding_lockup: An function to lockup embeddings for words indexes
        :return: Sampling string
        """

        y = Variable(t.zeros(1, self.params.word_embed_size))
        if z.is_cuda:
            y = y.cuda()
        y = t.cat([y, z], 1)

        hidden_state = self.initial_hidden_state(z)
        result = []

        for i in range(seq_len):
            y, hidden_state = self.unroll_cell(y, hidden_state)
            y = self.soft_argmax(y)

            word = batch_loader.decode_word(y.squeeze(0).data.cpu().numpy())
            result += [word]

            y = batch_loader.word_to_idx[word]
            y = Variable(t.from_numpy(np.array([y]))).long()
            y = embedding_lockup(y)
            y = t.cat([y, z], 1)

        return " ".join(result)

    def unroll_cell(self, input, hidden_state):
        for i, layer in enumerate(self.rnn):
            hidden_state[i] = layer(input, hidden_state[i])
            input = hidden_state[i]

        return self.hidden_to_vocab_size(input), hidden_state

    def initial_hidden_state(self, z):
        return [F.elu(mapping(z)) for mapping in self.latent_to_hidden]

    def learnable_parameters(self):
        return [par for par in self.parameters() if par.requires_grad]
