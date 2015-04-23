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

    def init_weights(self, batch_size):
        # Weights.
        self.weights = cm.CUDAMatrix(
            self.sigma * np.random.randn(self.size, self.next_size)
        )
        self.weights_transpose = \
            cm.empty((self.weights.shape[1], self.weights.shape[0]))
        self.weights_grad = cm.empty(self.weights.shape)

        # Bias.
        if self.use_bias:
            self.biases = cm.CUDAMatrix(
                self.sigma * np.random.randn(1, self.next_size)
            )
            self.active_biases = cm.empty(self.biases.shape)
            self.biases_grad = cm.empty(self.biases.shape)

        # Propagation.
        self.next_z = cm.empty((batch_size, self.next_size))
        self.a_transpose = cm.empty((self.size, batch_size))

        if self.level != 1:
            self.my_delta = cm.empty((batch_size, self.size))
        else:
            self.my_delta = None

    def forward_p(self, z):
        self.z = z
        self.activation_func.apply(self.z)
        cm.dot(self.z, self.weights, self.next_z)

        if self.use_bias:
            self.biases.mult(
                self.activation_func.apply_scalar(1),
                self.active_biases
            )
            self.next_z.add_row_vec(self.active_biases)
        return self.next_z

    def backward_p(self, next_delta):
        # Compute weights grad.
        self.z.transpose(self.a_transpose)
        cm.dot(self.a_transpose, next_delta,
               target=self.weights_grad)

        # Compute biases grad.
        if self.use_bias:
            next_delta.sum(0, self.biases_grad)

        if self.level != 1:
            self.weights.transpose(self.weights_transpose)
            cm.dot(next_delta, self.weights_transpose, self.my_delta)
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
