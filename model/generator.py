import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.functional import parameters_allocation_check
from torch_modules.other.highway import Highway
from torch_modules.other.gumbel_softmax import gumbel_softmax


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params

        self.rnn = nn.ModuleList([nn.GRUCell(input_size=size, hidden_size=self.params.gen_size[i + 1])
                                  for i, size in enumerate(self.params.gen_size[:-1])])

        self.highway = nn.ModuleList([Highway(size, 2, F.elu) for size in self.params.gen_size[1:]])

        self.fc = nn.Linear(self.params.gen_size[-1], self.params.vocab_size)

    def forward(self, x, z, seq_len, embedding_lockup):
        """
        :param x: An tensor with shape of [batch_size, word_embedding_size] filled with initial word in sequence
        :param z: An tensor with shape of [batch_size, latent_variable_size] to condition generation from 
        :param seq_len: length of generated sequence
        :param embedding_lockup: An function to lockup weighted embeddings
        :return: An tensor with shape of [batch_size, seq_len, vocab_size] 
                    containing probability disctribution over various words in vocabulary
        """

        [batch_size, latent_variable_size] = z.size()
        assert latent_variable_size == self.params.latent_variable_size, 'Invalid input size'

        '''Construct initial hidden state from latent variable and zero tensors'''
        hidden_state = [z] + [Variable(t.zeros(batch_size, size)) for size in self.params.gen_size[2:]]

        result = []

        for j in range(seq_len):
            x, hidden_state = self.unroll_cells(x, hidden_state)
            x = gumbel_softmax(x)
            x = embedding_lockup(x)

            result += [x]

        return t.cat([x.unsqueeze(1) for x in result], 1)

    def sample(self, x, z, seq_len, batch_loader, embedding_lockup):
        """
        :param x: An tensor with shape of [1, word_embedding_size] filled with initial word in sequence 
        :param z: An tensor with shape of [1, latent_variable_size] to condition generation from
        :param seq_len: length of generated sequence
        :param batch_loader: BatchLoader instance
        :return: An tensor with shape of [1, seq_len, vocab_size] 
                    containing probability disctribution over various words in vocabulary  
        """

        hidden_state = [z] + [Variable(t.zeros(1, size)) for size in self.params.gen_size[2:]]

        result = []

        for j in range(seq_len):
            x, hidden_state = self.unroll_cells(x, hidden_state)
            x = F.softmax(x)

            word = batch_loader.decode_word(x.squeeze(0).data.cpu().numpy())
            result += [word]

            x = batch_loader.word_to_idx[word]
            x = Variable(t.from_numpy(np.array([x]))).long()
            x = embedding_lockup(x)

        return ' '.join(result)

    def unroll_cells(self, x, hidden_state):
        """
        :param x: An tensor with shape of [batch_size, word_embedding_size] 
        :param hidden_state: An array of hidden states filled with tensors with shape of [batch_size, hidden_size_i]
        :return: Out of the last cell with shape of [batch_size, word_vocab_size]
                    and array of final states filled with tensors with shape of [batch_size, hidden_size_i]
        """

        for i, rnn_cell in enumerate(self.rnn):
            hidden_state[i] = rnn_cell(x, hidden_state[i])
            x = self.highway[i](hidden_state[i])

        x = self.fc(x)

        return x, hidden_state

    def learnable_parameters(self):
        return [par for par in self.parameters() if par.requires_grad]
