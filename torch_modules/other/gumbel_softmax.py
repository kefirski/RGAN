import torch as t
import torch.nn.functional as F
from torch.autograd import Variable


def _sample_gumbel(shape, eps=1e-20):
    unif = Variable(t.Tensor(*shape).uniform_(0, 1))
    return ((unif + eps).log().neg() + eps).log().neg()


def gumbel_softmax(logits, temperature=1e-1):
    """
    :param logits: An tensor with shape of [batch_size, input_size] 
    :param temperature: Non-negative scalar
    :return: An tensor with shape of [batch_size, input_size]  
    """

    '''
    Sample g ~ G(0, 1) and add it to logits
    Then evaluate differentiable reparametrization of distribution over the dims of input
    '''
    y = logits + _sample_gumbel(logits.size())
    y = F.softmax(y / temperature)

    return y
