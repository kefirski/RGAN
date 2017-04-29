import os
import numpy as np
import torch as t
import torch.nn as nn
from torch.nn import Parameter


class EmbeddingLockup(nn.Module):
    def __init__(self, params, path_prefix='../../../'):
        super(EmbeddingLockup, self).__init__()

        self.params = params

        word_embeddings_path = path_prefix + 'data/preprocessings/word_embeddings.npy'
        assert os.path.exists(word_embeddings_path), 'Word embeddings not found'

        embeddings = np.load(word_embeddings_path)

        self.embeddings = nn.Embedding(self.params.vocab_size, self.params.word_embed_size)
        self.embeddings.weight = Parameter(t.from_numpy(embeddings).float(), requires_grad=False)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size]
        """

        return self.embeddings(input)

    def weighted_lockup(self, input):
        """
        :param input: An 2D or 3D tensor with shape of [batch_size, Option(seq_len), vocab_size] 
        :return: An tensor with shape of [batch_size, Option(seq_len), word_embedding_size]
        """

        size = input.size()
        assert len(size) == 2 or len(size) == 3, 'Invalid input rang. 2D or 3D tensor is required'

        batch_size = size[0]

        if len(size) == 3:
            input = input.view(-1, self.params.vocab_size)

        result = t.mm(input, self.embeddings.weight)

        return result if len(size) == 2 else result.view(batch_size, -1, self.params.vocab_size)