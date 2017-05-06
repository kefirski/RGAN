import math
import torch
import torch.nn as nn
from torch.nn.modules.rnn import RNNCellBase
import torch.nn.functional as F
from torch.nn import Parameter
from torch_modules.layerNormGRUCell.layer_norm import LayerNormalization


class LayerNormGRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))

        self.reset_ln = LayerNormalization(self.hidden_size)
        self.input_ln = LayerNormalization(self.hidden_size)
        self.new_gate_ln = LayerNormalization(self.hidden_size)

        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):

        if input.is_cuda:
            gi = F.linear(input, self.weight_ih)
            gh = F.linear(hidden, self.weight_hh)
            state = fusedBackend.GRUFused()
            return state(gi, gh, hidden) if self.bias_ih is None else state(gi, gh, hidden, self.bias_ih, self.bias_hh)

        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden, self.weight_hh, self.bias_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(self.reset_ln(i_r + h_r))
        inputgate = F.sigmoid(self.input_ln(i_i + h_i))
        newgate = F.tanh(self.new_gate_ln(i_n + resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)

        return hy
