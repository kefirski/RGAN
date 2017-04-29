import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_modules.other.embedding_lockup import EmbeddingLockup
from torch.autograd import Variable
from .generator import Generator
from .discriminator import Discriminator
from utils.functional import parameters_allocation_check


class RGAN(nn.Module):
    def __init__(self, params, go_input, path_prefix=''):
        """
        :param params: utils.Parameters instance 
        :param path_prefix: path to data folder
        """

        super(RGAN, self).__init__()

        self.params = params

        self.embeddings = EmbeddingLockup(self.params, path_prefix)
        self.go_input = self.embeddings(go_input)

        self.generator = Generator(self.params)
        self.discriminator = Discriminator(self.params)

    def forward(self, z, true_data):
        """
        :param z: An tensor with shape of [batch_size, latent_variable_size] of uniform noise
        :param true_data: An tensor with shape of [batch_size, seq_len] of Long type
                              containing indexes of words of true data
        :return: discriminator and generator loss
        """

        [_, seq_len] = true_data.size()
        true_data = self.embeddings(true_data)

        generated_data = self.generator(self.go_input, z, seq_len, self.embeddings.weighted_lockup)

        return self.discriminator(generated_data, true_data)

    def sample(self, z, seq_len, batch_loader):
        """
        :param z: An tensor with shape of [1, latent_variable_size] of uniform noise
        :param seq_len: length of generated sequence
        :param batch_loader: BatchLoader instance
        :return: An tensor with shape of [1, seq_len, vocab_size] 
                    containing probability disctribution over various words in vocabulary        
        """

        go = self.go_input[0].unsqueeze(0)
        return self.generator.sample(go, z, seq_len, batch_loader, self.embeddings.forward)
