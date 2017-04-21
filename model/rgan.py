import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import parameters_allocation_check
from torch_modules.other.embedding_lockup import EmbeddingLockup
from .generator import Generator
from .discriminator import Discriminator


class RGAN(nn.Module):
    def __init__(self, params, path_prefix):
        """
        :param params: utils.Parameters instance 
        :param path_prefix: path to data folder
        """

        super(RGAN, self).__init__()

        self.params = params

        self.embedding = EmbeddingLockup(self.params, path_prefix)

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
        true_data = self.embedding(true_data)

        '''
        to make generated data continious it is necessary to sum up word embeddings
        with respect to their weights aka discrete probability disctribution values 
        '''
        generated_data = self.generator(z, seq_len)
        generated_data = self.embedding.weighted_lockup(generated_data)

        return self.discriminator(generated_data, true_data)



