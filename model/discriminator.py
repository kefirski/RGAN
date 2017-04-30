import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functional import parameters_allocation_check


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params

        self.rnn = nn.ModuleList(
            [nn.GRU(input_size=size, hidden_size=self.params.dis_size[i + 1], batch_first=True, bidirectional=True)
             for i, size in enumerate(self.params.dis_size[:-1])]
        )

        self.highway = nn.ModuleList([Highway(size, 2, F.elu) for size in self.params.dis_size[1:]])

        '''
        Since rnn is bidirectional shapes will be doubled during the forward propagation
        Thus it is necessary to map them back from these doubled dimensions
        '''
        self.retrieve_mapping = nn.ModuleList([nn.Linear(size * 2, size, F.elu) for size in self.params.dis_size[1:]])

        self.hidden_to_scalar = nn.Linear(self.params.dis_size[-1], 1)

    def forward(self, generated_data, true_data):
        """
        :param generated_data: An tensor with shape of [batch size, seq len, word embedding size] 
        :param true_data: An tensor with shape of [batch size, seq len, word embedding size]
        :return: Loss estimation for discriminator and generator in sense of Wasserstein GAN
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

        for i in range(len(self.rnn) - 1):
            x, _ = self.rnn[i](x)
            x = x.contiguous().view(batch_size * seq_len, -1)
            x = F.elu(self.retrieve_mapping[i](x))
            x = self.highway[i](x)
            x = x.view(batch_size, seq_len, -1)

        _, final_state = self.rnn[-1](x)
        final_state = final_state.transpose(0, 1) \
            .contiguous() \
            .view(-1, self.params.dis_size[-1] * 2)
        final_state = F.elu(self.retrieve_mapping[-1](final_state))
        final_state = self.highway[-1](final_state)

        return self.hidden_to_scalar(final_state).squeeze(1)



from torch_modules.other.highway import Highway
