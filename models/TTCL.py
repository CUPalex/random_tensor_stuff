import torch
from torch import nn
import numpy as np

import tensorly as tl

tl.set_backend('pytorch')

from tensorly import tenalg
from tltorch.factorized_tensors import FactorizedTensor, TuckerTensor, TTTensor
from tensorly.decomposition import tensor_train


class TTCL(nn.Module):
    def __init__(self, inp_modes, out_modes, kernel_size,
                 rank, p=0.9, stride=1, padding=0, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.p = p

        if len(inp_modes) != len(out_modes):
            raise ValueError(f'Something is wrong with the input and output modes. Got {inp_modes} and {out_modes}')

        if isinstance(inp_modes, int):
            self.inp_modes = (inp_modes,)
        else:
            self.inp_modes = tuple(inp_modes)

        if isinstance(out_modes, int):
            self.out_modes = (out_modes,)
        else:
            self.out_modes = tuple(out_modes)

        # initialize with decomposed normal distribution
        # if isinstance(kernel_size, int):
        #    kernel_size = (kernel_size, kernel_size)
        # full_weight = torch.normal(0.0, 0.02, size=kernel_size + tuple(np.array(self.inp_modes) * np.array(self.out_modes)))

        # factors = tensor_train(full_weight, rank=rank)
        #self.cores = torch.nn.ParameterList([torch.nn.Parameter(factor).to(self.device)
        #                                     for i, factor in enumerate(factors[2:])])

        #self.conv = nn.Conv2d(1, factors.rank[2], self.kernel_size,
        #                      stride=self.stride, padding=self.padding)
        #with torch.no_grad():
        #    g_0 = torch.einsum('aib,bjc->aijc', factors[0], factors[1])
        #    g_0 = torch.permute(g_0, (3, 0, 1, 2))
        #    self.conv.weight = nn.Parameter(g_0).to(self.device)

        self.rank = rank
        self.conv = nn.Conv2d(1, self.rank[0], self.kernel_size, stride=self.stride, padding=self.padding)
        self.cores = torch.nn.ParameterList([torch.nn.Parameter(nn.init.xavier_normal_(torch.empty((self.rank[i], self.inp_modes[i], self.out_modes[i], self.rank[i + 1])))).to(self.device)
                                             for i in range(len(self.inp_modes))])

        # self.rank = factors.rank[2:]

    def forward(self, x):
        # x_shape = [batch_size, c, h, w]
        # reshape x to [batch_size * c, 1, h, w]
        batch_size, c_in, h, w = tuple(x.shape)
        x = torch.reshape(x, (batch_size * c_in, 1, h, w))

        x = self.conv(x)

        # x_shape = [batch_size * c, rank[0], new_h, new_w]
        new_h, new_w = x.shape[-2:]
        x = torch.reshape(x, (batch_size, c_in, self.rank[0], new_h, new_w))
        x = torch.permute(x, (2, 1, 0, 3, 4))

        # x_shape = [rank[0], c_in, batch_size, new_h, new_w]

        # new_h, new_w = x.shape[-2:]
        d = len(self.out_modes)
        # x = torch.reshape(x, (batch_size, ) + self.inp_modes + (self.rank[0], new_h, new_w))
        # x = torch.permute(x, (d + 1, ) + tuple(np.arange(1, d + 1)) + (0, d + 2, d + 3))
        # x_shape = [rank[0], c_1, .. ,c_d, batch_size, new_h, new_w]

        for i in range(d):
            x = torch.reshape(x, (self.rank[i] * self.inp_modes[i], -1))
            if self.training:
                gamma = torch.Tensor(np.random.binomial(1, self.p, size=self.rank[i])).to(self.device)
                droppedout_core = (torch.nn.Parameter(torch.reshape(torch.einsum('i,ijkl->ijkl', gamma, self.cores[i]),
                       (self.rank[i] * self.inp_modes[i], self.rank[i + 1] * self.out_modes[i]))))
                x = torch.mm(droppedout_core.T, x)
            else:
                reshaped_core = (torch.nn.Parameter(torch.reshape(self.cores[i],
                       (self.rank[i] * self.inp_modes[i], self.rank[i + 1] * self.out_modes[i]))))
                x = torch.mm(reshaped_core.T, x)
                # x = torch.einsum('kc...bhw,kcoj->j...obhw', x, self.cores[i])
            x = torch.reshape(x, (self.rank[i + 1], self.out_modes[i], -1))
            x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (batch_size, new_h, new_w, int(torch.prod(torch.tensor(self.out_modes)).item())))
        x = torch.permute(x, (0, 3, 1, 2))
        return x
