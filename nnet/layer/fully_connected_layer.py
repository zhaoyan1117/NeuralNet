from __future__ import absolute_import, division

from .base import LayerBase

import numpy as np

class FullyConnectedLayer(LayerBase):

    def __init__(self, size, next_size, activation_func, depth, bias=True):
        self.depth = depth
        self.size = size
        self.next_size = next_size
        self.activation_func = activation_func
        self.bias = bias
        self._init_weights()

    def forward(self, z):
        if self.bias:
            bias_term = np.ones((len(z), 1))
            z = np.append(z, bias_term, axis=1)

        a = self.activation_func.apply(z)
        next_z = np.dot(a, self.weights)
        return next_z

    def _init_weights(self):
        if self.bias:
            self.weights = np.random.randn(self.size+1, self.next_size)
        else:
            self.weights = np.random.randn(self.size, self.next_size)
