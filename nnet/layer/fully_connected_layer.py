from __future__ import absolute_import, division

import numpy as np

from .base import LayerBase

class FullyConnectedLayer(LayerBase):

    def __init__(self, size, next_size, activation_func, level,
                 bias=True, sigma=1.0):
        self.level = level
        self.size = size
        self.next_size = next_size
        self.activation_func = activation_func
        self.bias = bias
        self.a = None
        self.z = None
        self.delta = None
        self.weights_derivative = None
        self._init_weights(sigma)

    def _init_weights(self, sigma):
        if self.bias:
            self.weights = sigma * np.random.randn(self.size+1, self.next_size)
        else:
            self.weights = sigma * np.random.randn(self.size, self.next_size)

    def forward_p(self, z):
        if self.bias:
            bias_term = np.ones((len(z), 1))
            self.z = np.append(z, bias_term, axis=1)
        else:
            self.z = z

        self.a = self.activation_func.apply(z)
        next_z = np.dot(self.a, self.weights)
        return next_z

    def backward_p(self, next_delta):
        self.weights_derivative = np.dot(self.a.T, next_delta)

        # No need to compute delta if layer is the first layer.
        if self.level != 1:
            self.delta = np.dot(next_delta, self.weights.T) \
                         * self.activation_func.apply_derivative(self.z)

        return self.delta

    def update(self, lr):
        self.weights = self.weights - lr * self.weights_derivative
