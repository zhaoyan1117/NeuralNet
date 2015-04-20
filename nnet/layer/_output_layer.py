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
        del self.z
        del self.a

        self.z = z
        self.a = self.activation_func.apply(self.z)
        return self.a

    def backward_p(self, y):
        my_delta = cm.empty(y.shape)
        self.a.subtract(y, my_delta)
        my_delta.divide(float(my_delta.shape[0]))

        if self.loss_func == 'MSE':
            my_delta.mult(self.activation_func.apply_derivative(self.z))
        elif self.loss_func == 'CEE':
            pass
        else:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))

        return my_delta

    def compute_loss(self, y):
        if self.loss_func == 'MSE':
            return self._compute_loss_MSE(y)
        elif self.loss_func == 'CEE':
            return self._compute_loss_CEE(y)
        else:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))

    def _compute_loss_MSE(self, y):
        diff = cm.empty(y.shape)
        y.subtract(self.a, diff)
        cm.pow(diff, 2, diff)
        diff_sum = self._find_sum(diff)
        size = diff.shape[0]
        del diff
        return diff_sum / float(2*size)

    def _compute_loss_CEE(self, y):
        # Copy to cpu to compute loss due to numerical issue.
        # This should be a huge performance bottleneck
        # since we don't compute loss at every iteration.
        cpu_y = y.asarray().astype(np.double)
        cpu_y_hat = self.a.asarray().astype(np.double)

        cpu_y_hat[np.nonzero(cpu_y_hat==0)] = 1e-8
        cpu_y_hat[np.nonzero(cpu_y_hat==1)] = 1-1e-8

        entropy = cpu_y * np.log(cpu_y_hat) \
                  + (1.0 - cpu_y) * np.log(1.0 - cpu_y_hat)
        return -np.sum(entropy) \
               / float(len(entropy))

    def _find_sum(self, mat):
        col_sum = mat.sum(axis=0)
        row_sum = col_sum.sum(axis=1)
        sum = row_sum.asarray()[0][0]
        del col_sum
        del row_sum
        return sum

    def update(self, lr):
        # No weights to update for output layer.
        return None
