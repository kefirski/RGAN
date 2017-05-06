import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SoftArgmax(nn.Module):
    def __init__(self, temperature=1e-3):
        super(SoftArgmax, self).__init__()
        self.temperature = temperature

    def forward(self, input, sampling=False):
        size = input.size()

        if sampling:
            noise = SoftArgmax._sample_gumbel(size)
            input = input + noise

        input = input.view(-1, size[-1])
        input = F.softmax(input / self.temperature)

        return input.view(*size)

    @staticmethod
    def _sample_gumbel(shape, eps=1e-20):
        unif = Variable(t.Tensor(*shape).uniform_(0, 1))
        return ((unif + eps).log().neg() + eps).log().neg()