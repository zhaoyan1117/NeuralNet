from __future__ import absolute_import, division

import numpy as np
import cudamat as cm

from ._base import LayerBase


class FullyConnectedLayer(LayerBase):

    def __init__(self, level, size, activation_func, sigma=1.0, bias=True):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.bias = bias
        self.sigma = sigma

    def set_next_layer_size(self, next_size):
        self.next_size = next_size
        self._init_weights()

    def _init_weights(self):
        self.a = None
        self.z = None

        # Re-use weights and weights_grad.
        actual_size = self.size+1 if self.bias else self.size
        self.weights = cm.CUDAMatrix(self.sigma * np.random.randn(actual_size, self.next_size))
        self.weights_grad = cm.empty(self.weights.shape)

    def forward_p(self, z):
        self._free_mem()

        if self.bias:
            self.z = cm.CUDAMatrix(np.ones((z.shape[0], z.shape[1]+1)))
            self.z.set_col_slice(0, z.shape[1], z)
        else:
            self.z = z

        self.a = self.activation_func.apply(self.z)
        self.next_z = cm.dot(self.a, self.weights)
        return self.next_z

    def backward_p(self, next_delta):
        cm.dot(self.a.transpose(), next_delta, target=self.weights_grad)

        self.my_delta = None
        if self.level != 1:
            temp_delta = cm.dot(next_delta, self.weights.transpose())\
                .mult(self.activation_func.apply_derivative(self.z))
            if self.bias:
                row, col = temp_delta.shape
                self.my_delta = temp_delta.get_col_slice(0, col-1)
            else:
                self.my_delta = temp_delta

        return self.my_delta

    def update(self, lr):
        self.weights.subtract(self.weights_grad.mult(lr))
