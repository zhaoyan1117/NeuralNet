from __future__ import absolute_import, division

import numpy as np

from ._base import LayerBase

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
        self.weights_derivative = None

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

    def numerical_check(self, net):
        epsilon = 1e-5
        current_weights = self.weights
        num_grad = np.zeros(self.weights.shape)
        perturb = np.zeros(self.weights.shape)

        for i in xrange(len(current_weights)):
            for j in xrange(len(current_weights[i])):
                perturb[i][j] = epsilon

                self.weights = current_weights - perturb
                loss1 = net.compute_all_loss()

                self.weights = current_weights + perturb
                loss2 = net.compute_all_loss()

                num_grad[i][j] = (loss2 - loss1) / (2*epsilon)

                perturb[i][j] = 0.0

        self.weights = current_weights

        diff = (self.weights_derivative - num_grad).ravel()

        return np.linalg.norm(diff) < 1e-8
