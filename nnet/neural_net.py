from __future__ import absolute_import

from .neural_net_exception import NeuralNetException

class NeuralNet(object):

    def __init__(self, batch_size, lr_func, loss_func, layers, loss_period):
        self.batch_size = batch_size
        self.lr_func = lr_func
        self.loss_func = loss_func
        self.layers = layers
        self.data, self.labels = None, None
        self.loss_period = loss_period

    def train(self, data, labels):
        pass

    def forward_p(self, data):
        cur_z = data
        for l in self.layers:
            cur_z = l.forward_p(cur_z)
        return cur_z

    def backward_p(self, delta):
        cur_delta = delta
        for l in reversed(self.layers):
            cur_delta = l.backward_p(cur_delta)

    def update(self, lr):
        for l in self.layers:
            l.update(lr)

    def numerical_check(self):
        for l in self.layers:
            passed = l.numerical_check(self)
            if not passed:
                print "WARNING: layer {0} " \
                      "failed numerical check.".format(l.level)

    def compute_loss(self, labels, predictions):
        return self.loss_func.apply(labels, predictions)

    def compute_all_loss(self):
        if self.data is None:
            raise NeuralNetException('Cannot compute loss without data.')

        return self.compute_loss(self.labels,
                                 self.forward_p(self.data))
