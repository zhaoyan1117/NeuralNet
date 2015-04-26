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

        self.use_momentum = kwargs.get('use_momentum', False)
        if self.use_momentum:
            self.momentum = kwargs.get('momentum', 0.9)

        self.use_dropout = kwargs.get('use_dropout', False)
        if self.use_dropout:
            self.dropout_p = kwargs.get('dropout_p', 0.5)
        self.use_max_norm = kwargs.get('use_max_norm', False)
        if self.use_max_norm:
            self.max_norm_c = kwargs.get('max_norm_c', 4.0)

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

        # Max norm params.
        if self.use_max_norm:
            self._init_max_norm_params()

    def _init_weights(self):
        var = math.sqrt(2.0/self.size) if self.sigma == 'c' else self.sigma
        self.weights = cm.CUDAMatrix(
            var * np.random.randn(self.size, self.next_size)
        )

        self.weights_grad = cm.empty(self.weights.shape)

        if self.use_momentum:
            self.weights_update = \
                cm.CUDAMatrix(np.zeros(self.weights_grad.shape))

    def _init_bias(self):
        assert self.use_bias
        self.biases = cm.CUDAMatrix(
            np.zeros((1, self.next_size))
        )
        self.active_biases = cm.empty(self.biases.shape)
        self.biases_grad = cm.empty(self.biases.shape)

        if self.use_momentum:
            self.biases_update = \
                cm.CUDAMatrix(np.zeros(self.biases_grad.shape))

    def _init_params(self, batch_size):
        self.next_z = cm.empty((batch_size, self.next_size))

        if self.level != 1:
            self.my_delta = cm.empty((batch_size, self.size))
        else:
            self.my_delta = None

    def _init_dropout_mask(self, batch_size):
        assert self.use_dropout
        self.dropout_mask = cm.empty((batch_size, self.size))

    def _init_max_norm_params(self):
        assert self.use_max_norm
        self.weights_square = cm.empty(self.weights.shape)
        self.weights_factor = cm.empty((1, self.next_size))
        self.weights_factor_mask = cm.empty((1, self.next_size))

    def forward_p(self, z, predict=False):
        self.z = z
        self.activation_func.apply(self.z)

        # Dropout regularization.
        if self.use_dropout and (not predict):
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
        cm.dot(self.z.T, next_delta, self.weights_grad)

        # Compute biases grad.
        if self.use_bias:
            next_delta.sum(0, self.biases_grad)

        if self.level != 1:
            cm.dot(next_delta, self.weights.T, self.my_delta)
            self.activation_func.mult_with_derivative(self.my_delta, self.z)

        return self.my_delta

    def update(self, lr):
        if self.use_momentum:
            self.weights_update.mult(self.momentum)
            self.weights_update.subtract_mult(self.weights_grad, lr)
            self.weights.add(self.weights_update)

            if self.use_bias:
                self.biases_update.mult(self.momentum)
                self.biases_update.subtract_mult(self.biases_grad, lr)
                self.biases.add(self.biases_update)
        else:
            self.weights.subtract_mult(self.weights_grad, lr)
            if self.use_bias:
                self.biases.subtract_mult(self.biases_grad, lr)

        # Max-norm regularization.
        if self.use_max_norm:
            cm.pow(self.weights, 2, self.weights_square)
            self.weights_square.sum(0, self.weights_factor)
            cm.sqrt(self.weights_factor, self.weights_factor)

            # Avoid zero weight mags.
            self.weights_factor.add(1e-8)
            self.weights_factor.reciprocal().mult(self.max_norm_c)

            # Filter not factor greater than 1.0
            self.weights_factor.less_than(1.0, self.weights_factor_mask)
            self.weights_factor.mult(self.weights_factor_mask)

            # Change 0.0 entry to 1.0.
            self.weights_factor_mask.less_than(1.0)
            self.weights_factor.add(self.weights_factor_mask)

            # Down scale over sized weights.
            self.weights.mult_by_row(self.weights_factor)

    def dump_params(self):
        del self.z

        # Weights.
        self._dump_np('weights')
        self._dump_np('weights_grad')
        if self.use_momentum:
            self._dump_np('weights_update')

        # Biases.
        if self.use_bias:
            self._dump_np('biases')
            self._dump_np('active_biases')
            self._dump_np('biases_grad')
            if self.use_momentum:
                self._dump_np('biases_update')

        # Params.
        self._dump_np('next_z')
        if self.level != 1:
            self._dump_np('my_delta')

        # Dropout mask.
        if self.use_dropout:
            self._dump_np('dropout_mask')

        # Max-norm.
        if self.use_max_norm:
            self._dump_np('weights_square')
            self._dump_np('weights_factor')
            self._dump_np('weights_factor_mask')

    def load_params(self):
        # Weights.
        self._load_np('weights')
        self._load_np('weights_grad')
        if self.use_momentum:
            self._load_np('weights_update')

        # Biases.
        if self.use_bias:
            self._load_np('biases')
            self._load_np('active_biases')
            self._load_np('biases_grad')
            if self.use_momentum:
                self._load_np('biases_update')

        # Params.
        self._load_np('next_z')
        if self.level != 1:
            self._load_np('my_delta')

        # Dropout mask.
        if self.use_dropout:
            self._load_np('dropout_mask')

        # Max-norm.
        if self.use_max_norm:
            self._load_np('weights_square')
            self._load_np('weights_factor')
            self._load_np('weights_factor_mask')
