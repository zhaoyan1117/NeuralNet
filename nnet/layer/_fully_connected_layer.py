from __future__ import absolute_import, division

import numpy as np
import cudamat as cm

from ._base import LayerBase


class FullyConnectedLayer(LayerBase):

    def __init__(self, level, size, activation_func, sigma=1.0, use_bias=True):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.use_bias = use_bias
        self.sigma = sigma

    def set_next_layer_size(self, next_size):
        self.next_size = next_size
        self._init_weights()

    def _init_weights(self):
        self.weights = cm.CUDAMatrix(
            self.sigma * np.random.randn(self.size, self.next_size)
        )
        self.weights_transpose = \
            cm.empty((self.weights.shape[1], self.weights.shape[0]))
        self.weights_grad = cm.empty(self.weights.shape)

        if self.use_bias:
            self.biases = cm.CUDAMatrix(
                self.sigma * np.random.randn(1, self.next_size)
            )
            self.active_biases = cm.empty(self.biases.shape)
            self.biases_grad = cm.empty(self.biases.shape)

    def forward_p(self, z):
        self._free_mem()

        self.z = z
        self.activation_func.apply(self.z)
        self.next_z = cm.dot(self.z, self.weights)

        if self.use_bias:
            self.biases.mult(
                self.activation_func.apply_scalar(1),
                self.active_biases
            )
            self.next_z.add_row_vec(self.active_biases)
        return self.next_z

    def backward_p(self, next_delta):
        # Compute weights grad.
        a_transpose = self.z.transpose()
        cm.dot(a_transpose, next_delta, target=self.weights_grad)
        a_transpose.free_device_memory()
        del a_transpose

        # Compute biases grad.
        if self.use_bias:
            next_delta.sum(0, self.biases_grad)

        self.my_delta = None

        if self.level != 1:
            self.weights.transpose(self.weights_transpose)
            self.my_delta = cm.dot(next_delta, self.weights_transpose)
            self.activation_func.mult_with_derivative(self.my_delta, self.z)

        return self.my_delta

    def update(self, lr):
        self.weights.subtract_mult(self.weights_grad, lr)
        if self.use_bias:
            self.biases.subtract_mult(self.biases_grad, lr)

    def predict(self, z):
        self.activation_func.apply(z)
        self.predict_z = cm.dot(z, self.weights)

        if self.use_bias:
            self.biases.mult(
                self.activation_func.apply_scalar(1),
                self.active_biases
            )
            self.predict_z.add_row_vec(self.active_biases)
        return self.predict_z
