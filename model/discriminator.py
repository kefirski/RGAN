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

    def forward(self, generated_data, true_data):
        """
        :param generated_data: An tensor with shape of [batch size, seq len, word embedding size] 
        :param true_data: An tensor with shape of [batch size, seq len, word embedding size]
        :return: Loss estimation for discriminator and generator in sense of Wasserstein GAN
        """

        generated_labels = self.unroll_network(generated_data)
        true_labels = self.unroll_network(true_data)

        discriminator_loss = -true_labels.mean() + generated_labels.mean()
        generator_loss = -generated_labels.mean()

        return discriminator_loss, generator_loss

    def unroll_network(self, input):
        """
        :param input: An tensor with shape of [batch size, seq len, word embedding size] 
        :return: Last hidden state of rnn network passed through all layers with shape of [batch_size]
        """

        for i, layer in enumerate(self.rnn):
            input, _ = layer(input)

            # the last layer of rnn emits last hidden state - thus it is no necessary to perform highway on whole output
            if i != len(self.rnn) - 1:
                input = self.batch_norm[i](input)
                input = self.highway[i](input)

            _, last_state = self.rnn[-1](input)
            last_state = self.batch_norm[-1](last_state)
            last_state = self.highway[-1](last_state)

            return self.fc(last_state).squeeze(1)
