from __future__ import absolute_import, division

import numpy as np

from ._base import LayerBase
from ..util import iterate_with_progress

class FullyConnectedLayer(LayerBase):

    def __init__(self, level, size, activation_func, sigma=1.0, bias=True):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.bias = bias
        self.sigma = sigma
        self.a = None
        self.z = None
        self.delta = None
        self.weights_grad = None

    def set_next_layer_size(self, next_size):
        self.next_size = next_size
        self._init_weights(self.sigma)

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

        self.a = self.activation_func.apply(self.z)
        next_z = np.dot(self.a, self.weights)
        return next_z

    def backward_p(self, next_delta):
        self.weights_grad = np.dot(self.a.T, next_delta)

        # No need to compute delta if layer is the first layer.
        if self.level != 1:
            self.delta = np.dot(next_delta, self.weights.T) \
                         * self.activation_func.apply_derivative(self.z)
            if self.bias:
                self.delta = np.delete(self.delta, -1, axis=1)

        return self.delta

    def update(self, lr):
        self.weights = self.weights - lr * self.weights_grad

    def numerical_check(self, net):
        epsilon = 1e-5
        current_weights = self.weights
        num_grad = np.zeros(self.weights.shape)
        perturb = np.zeros(self.weights.shape)

        total_size = current_weights.shape[0] * current_weights.shape[1]

        for k in iterate_with_progress(xrange(total_size)):
            i = k % current_weights.shape[0]
            j = k // current_weights.shape[0]

            perturb[i][j] = epsilon

            self.weights = current_weights - perturb
            loss1 = net.compute_all_loss()

            self.weights = current_weights + perturb
            loss2 = net.compute_all_loss()

            num_grad[i][j] = (loss2 - loss1) / (2*epsilon)
            # print self.weights_grad[i][j] - num_grad[i][j]

            perturb[i][j] = 0.0

        self.weights = current_weights

        diff = np.linalg.norm((self.weights_grad - num_grad).ravel())
        sum = np.linalg.norm((self.weights_grad + num_grad).ravel())
        return diff/sum < 1e-8
