from __future__ import absolute_import, division

import cudamat as cm
import numpy as np

from ._base import LayerBase
from .._neural_net_exception import NeuralNetException


class OutputLayer(LayerBase):
    SUPPORTED_LOSS_FUNC = [
        'MSE',
        'CEE'
    ]

    def __init__(self, level, size, activation_func, loss_func):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.loss_func = loss_func
        if self.loss_func not in OutputLayer.SUPPORTED_LOSS_FUNC:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))
        self.z = None
        self.a = None

    def set_next_layer_size(self, next_size):
        # Output layer does not have next layer.
        pass

    def forward_p(self, z):
        self._free_mem()

        self.z = z
        self.a = self.activation_func.apply(self.z)
        return self.a

    def backward_p(self, y):
        self.my_delta = cm.empty(y.shape)
        self.a.subtract(y, self.my_delta)
        self.my_delta.divide(float(self.my_delta.shape[0]))

        if self.loss_func == 'MSE':
            self.my_delta.mult(self.activation_func.apply_derivative(self.z))
        elif self.loss_func == 'CEE':
            pass
        else:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))

        return self.my_delta

    def compute_loss(self, y):
        if self.loss_func == 'MSE':
            return self._compute_loss_MSE(y)
        elif self.loss_func == 'CEE':
            return self._compute_loss_CEE(y)
        else:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))

    def _compute_loss_MSE(self, y):
        # Copy to cpu to compute loss due to numerical issue.
        # This should not be a huge performance bottleneck
        # since we don't compute loss at every iteration.
        cpu_y = y.asarray().astype(np.double)
        cpu_y_hat = self.a.asarray().astype(np.double)
        diff = cpu_y - cpu_y_hat
        return np.sum(diff**2) \
               / float(2*len(diff))

    def _compute_loss_CEE(self, y):
        # Copy to cpu to compute loss due to numerical issue.
        # This should not be a huge performance bottleneck
        # since we don't compute loss at every iteration.
        cpu_y = y.asarray().astype(np.double)
        cpu_y_hat = self.a.asarray().astype(np.double)

        cpu_y_hat[np.nonzero(cpu_y_hat==0)] = 1e-8
        cpu_y_hat[np.nonzero(cpu_y_hat==1)] = 1-1e-8

        entropy = cpu_y * np.log(cpu_y_hat) \
                  + (1.0 - cpu_y) * np.log(1.0 - cpu_y_hat)
        return -np.sum(entropy) \
               / float(len(entropy))

    def update(self, lr):
        # No weights to update for output layer.
        return None
