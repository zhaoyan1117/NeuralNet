from __future__ import absolute_import

from .neural_net_exception import NeuralNetException

class NeuralNet(object):

    def __init__(self, batch_size, lr_func, loss_func, layers):
        self.batch_size = batch_size
        self.lr_func = lr_func
        self.loss_func = loss_func
        self.layers = layers
        self.data, self.labels = None, None

    def train(self, data, labels):
        pass

    def forward_p(self):
        pass

    def backward_p(self):
        pass

    def numerical_check(self):
        pass

    def compute_loss(self, data, labels):
        pass

    def compute_all_loss(self):
        if self.data is None:
            raise NeuralNetException('Cannot compute loss without data.')
        return self.compute_loss(self.data, self.labels)
