import torch
from torch import nn
import numpy as np

import tensorly as tl
tl.set_backend('pytorch')

from tensorly import tenalg
from tltorch.factorized_tensors import FactorizedTensor, TuckerTensor, TTTensor
from tensorly.decomposition import tensor_train

class TTRL(nn.Module):
    def __init__(self, input_shape, output_shape, bias=False, verbose=0,
                 rank='same', p=0.9, device=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.device = device

        if len(input_shape) != len(output_shape):
            raise ValueError(
                f'input and output shape lenghts must be the same. got {input_shape} input_shape and {output_shape} output_shape')

        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        else:
            self.input_shape = tuple(input_shape)

        if isinstance(output_shape, int):
            self.output_shape = (output_shape,)
        else:
            self.output_shape = tuple(output_shape)

        self.n_input = len(self.input_shape)
        self.n_output = len(self.output_shape)
        self.weight_shape = []
        for i in range(len(self.input_shape)):
            self.weight_shape.extend([self.input_shape[i], self.output_shape[i]])
        self.order = len(self.weight_shape)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_shape, device=device, dtype=dtype))
        else:
            self.bias = self.register_parameter("bias", None)

        # initialize with decomposed normal distribution
        full_weight = torch.normal(0.0, 0.02, size=tuple(np.array(input_shape) * np.array(output_shape)))

        factors = tensor_train(full_weight, rank=rank)
        self.rank = factors.rank
        self.factors = torch.nn.ParameterList([torch.nn.Parameter(factor).to(self.device) \
                                               for i, factor in enumerate(factors)])
        self.factorization = 'tt'
        self.p = p

    def forward(self, x):
        return self.tt_rtrl(x).reshape((x.shape[0],) + (np.prod(self.output_shape),))

    def tt_rtrl(self, x, p=0.9):
        x = x.reshape((x.shape[0],) + self.input_shape)
        n_input = tl.ndim(x) - 1

        if self.training:
            droppedout_factors = []
            gammas = [torch.Tensor(np.random.binomial(1, self.p, size=r)).to(self.device) for r in self.rank[1:-1]]

            new_tt = FactorizedTensor.new(shape=tuple(np.array(self.input_shape) * np.array(self.output_shape)), \
                                          rank=self.rank, factorization='TT')
            for i, factor in enumerate(self.factors[:-1]):
                new_tt.factors[i] = (torch.nn.Parameter(torch.einsum('ijk,k->ijk', factor, gammas[i])))
            new_tt.factors[-1] = torch.nn.Parameter(self.factors[-1])
            regression_weights = new_tt.to_tensor().reshape(self.weight_shape)
            regression_weights = torch.permute(regression_weights, \
                                               tuple(np.hstack((np.arange(0, len(self.weight_shape), 2), \
                                                                np.arange(1, len(self.weight_shape), 2)))))
        else:
            new_tt = FactorizedTensor.new(shape=tuple(np.array(self.input_shape) * np.array(self.output_shape)), \
                                          rank=self.rank, factorization='TT')
            for i, factor in enumerate(self.factors):
                new_tt.factors[i] = torch.nn.Parameter(factor.to(self.device))
            regression_weights = new_tt.to_tensor().reshape(self.weight_shape)
            regression_weights = torch.permute(regression_weights, \
                                               tuple(np.hstack((np.arange(0, len(self.weight_shape), 2), \
                                                                np.arange(1, len(self.weight_shape), 2)))))

        if self.bias is None:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x) - 1)
        else:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x) - 1) + self.bias