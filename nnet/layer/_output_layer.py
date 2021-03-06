from __future__ import absolute_import, division

import cudamat as cm
import numpy as np

from ._base import LayerBase
from .._neural_net_exception import NeuralNetException
import nnet.activation_func as af


class OutputLayer(LayerBase):
    SUPPORTED_LOSS_FUNC = [
        'MSE',
        'CEE'
    ]

    SUPPORTED_CEE_ACT_FUNC = [
        af.Sigmoid
    ]

    def __init__(self, level, size, activation_func, loss_func):
        self.level = level
        self.size = size

        self.loss_func = loss_func
        if self.loss_func not in OutputLayer.SUPPORTED_LOSS_FUNC:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))

        self.activation_func = activation_func
        if self.loss_func == 'CEE' \
                and (not any([type(activation_func) == f for f in OutputLayer.SUPPORTED_CEE_ACT_FUNC])):
            raise NeuralNetException('Activation func {0} is not supported '
                                     'with loss func CEE.'.format(self.activation_func))

    def set_next_layer_size(self, next_size):
        # Output layer does not have next layer.
        pass

    def init(self, batch_size):
        self.batch_size = batch_size
        self.my_delta = cm.empty((batch_size, self.size))

    def forward_p(self, z, predict=False):
        self.z = z
        self.activation_func.apply(self.z)
        return self.z

    def forward_p_single(self, single_z):
        return self.forward_p(single_z, True)

    def backward_p(self, y):
        self.z.subtract(y, self.my_delta)
        self.my_delta.divide(float(self.my_delta.shape[0]))

        if self.loss_func == 'MSE':
            self.activation_func\
                .mult_with_derivative(self.my_delta, self.z)
        elif self.loss_func == 'CEE':
            # Currently only support Sigmoid as loss function
            # for output layer with loss function CEE.
            pass
        else:
            raise NeuralNetException('Loss func {0} is not supported.'.format(self.loss_func))

        return self.my_delta

    def update(self, lr):
        # No weights to update for output layer.
        pass

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
        cpu_y_hat = self.z.asarray().astype(np.double)
        diff = cpu_y - cpu_y_hat
        return np.sum(diff**2) \
               / float(2*self.batch_size)

    def _compute_loss_CEE(self, y):
        # Copy to cpu to compute loss due to numerical issue.
        # This should not be a huge performance bottleneck
        # since we don't compute loss at every iteration.
        cpu_y = y.asarray().astype(np.double)
        cpu_y_hat = self.z.asarray().astype(np.double)

        cpu_y_hat[np.nonzero(cpu_y_hat==0)] = 1e-8
        cpu_y_hat[np.nonzero(cpu_y_hat==1)] = 1-1e-8

        entropy = cpu_y * np.log(cpu_y_hat) \
                  + (1.0 - cpu_y) * np.log(1.0 - cpu_y_hat)
        return -np.sum(entropy) \
               / float(self.batch_size)

    def dump_params(self):
        del self.z
        self._dump_np('my_delta')

    def load_params(self):
        self._load_np('my_delta')
