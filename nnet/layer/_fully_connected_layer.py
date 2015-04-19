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
        # Throw-away a, z, and delta.
        self.a = None
        self.z = None
        self.delta = None

        actual_size = self.size+1 if self.bias else self.size

        # Re-use weights and weights_grad.
        self.weights = cm.CUDAMatrix(self.sigma * np.random.randn(actual_size, self.next_size))
        self.weights_grad = cm.empty(self.weights.shape)

    def forward_p(self, z):
        del self.z
        del self.a

        if self.bias:
            # Copy back to cpu to append.
            cpu_z = z.asarray()
            bias_term = np.ones((len(cpu_z), 1))
            self.z = cm.CUDAMatrix(np.append(cpu_z, bias_term, axis=1))
        else:
            self.z = z

        self.a = self.activation_func.apply(self.z)
        next_z = cm.dot(self.a, self.weights)
        return next_z

    def backward_p(self, next_delta):
        del self.delta

        cm.dot(self.a.transpose(), next_delta, target=self.weights_grad)

        # No need to compute delta if layer is the first layer.
        if self.level != 1:
            temp_delta = cm.dot(next_delta, self.weights.transpose())\
                .mult(self.activation_func.apply_derivative(self.z))

            if self.bias:
                row, col = temp_delta.shape
                self.delta = cm.empty((row, col-1))
                temp_delta.get_col_slice(0, col-1, self.delta)
                del temp_delta
            else:
                self.delta = temp_delta

        return self.delta

    def update(self, lr):
        self.weights.subtract(self.weights_grad.mult(lr))
