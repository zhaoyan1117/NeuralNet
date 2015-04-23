from __future__ import absolute_import, division

import math

import numpy as np
import cudamat as cm

from ._base import LayerBase


class FullyConnectedLayer(LayerBase):

    def __init__(self, level, size, activation_func,
                 sigma='c', use_bias=True, **kwargs):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.use_bias = use_bias
        self.sigma = sigma

        # Regularization techniques.
        self.use_dropout = kwargs.get('use_dropout', False)
        if self.use_dropout:
            self.dropout_p = kwargs.get('dropout_p', 0.5)

    def set_next_layer_size(self, next_size):
        self.next_size = next_size

    def init(self, batch_size):
        # Weights.
        self._init_weights()

        # Bias.
        if self.use_bias:
            self._init_bias()

        # Propagation.
        self._init_params(batch_size)

        # Dropout mask.
        if self.use_dropout:
            self._init_dropout_mask(batch_size)

    def _init_weights(self):
        var = math.sqrt(2.0/self.size) if self.sigma == 'c' else self.sigma
        self.weights = cm.CUDAMatrix(
            var * np.random.randn(self.size, self.next_size)
        )

        self.weights_transpose = \
            cm.empty((self.weights.shape[1], self.weights.shape[0]))

        self.weights_grad = cm.empty(self.weights.shape)

    def _init_bias(self):
        assert self.use_bias
        self.biases = cm.CUDAMatrix(
            np.zeros((1, self.next_size))
        )
        self.active_biases = cm.empty(self.biases.shape)
        self.biases_grad = cm.empty(self.biases.shape)

    def _init_params(self, batch_size):
        self.next_z = cm.empty((batch_size, self.next_size))
        self.a_transpose = cm.empty((self.size, batch_size))

        if self.level != 1:
            self.my_delta = cm.empty((batch_size, self.size))
        else:
            self.my_delta = None

    def _init_dropout_mask(self, batch_size):
        self.dropout_mask = cm.empty((batch_size, self.size))

    def forward_p(self, z):
        self.z = z
        self.activation_func.apply(self.z)

        # Dropout regularization.
        if self.use_dropout:
            self.dropout_mask\
                .fill_with_rand()\
                .less_than(self.dropout_p)\
                .divide(self.dropout_p)
            self.z.mult(self.dropout_mask)

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
        cm.dot(self.a_transpose, next_delta, self.weights_grad)

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
