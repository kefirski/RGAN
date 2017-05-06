import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.highway import Highway
from utils.functional import parameters_allocation_check
from torch_modules.layerNormGRUCell.layerNormGRUCell import LayerNormGRUCell
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params

        self.rnn = nn.ModuleList([LayerNormGRUCell(input_size=size, hidden_size=self.params.dis_size[i + 1])
                                  for i, size in enumerate(self.params.dis_size[:-1])])

        self.hidden_to_scalar = nn.Linear(self.params.dis_size[-1], 1)

    def forward(self, generated_data, true_data):
        """
        :param generated_data: An tensor with shape of [batch size, seq len, word embedding size] 
        :param true_data: An tensor with shape of [batch size, seq len, word embedding size]
        :return: Loss estimation for discriminator and generator
        """

        true_labels = self.unroll_network(true_data)
        generated_labels = self.unroll_network(generated_data)

        discriminator_loss = -true_labels.mean() + generated_labels.mean()
        generator_loss = -generated_labels.mean()

        return discriminator_loss, generator_loss

    def unroll_network(self, x):
        """
        :param x: An tensor with shape of [batch_size, seq_len, word_embedding_size] 
        :return: Last hidden state of discriminator network passed through all layers with shape of [batch_size]
        """

        [batch_size, seq_len, _] = x.size()

        result = x.chunk(dim=1, num_chunks=seq_len)
        result = [chunk.squeeze(1) for chunk in result]

        for i, layer in enumerate(self.rnn):
            hidden_state = Variable(t.zeros(batch_size, layer.hidden_size))
            if x.is_cuda:
                hidden_state = hidden_state.cuda()

            for j, input in enumerate(result):
                hidden_state = layer(input, hidden_state)
                result[j] = hidden_state

        return self.hidden_to_scalar(result[-1]).squeeze(1)
