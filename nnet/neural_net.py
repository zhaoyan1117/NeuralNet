from __future__ import absolute_import

class NeuralNet(object):

    def __init__(self, lr_func, loss_func, layers):
        self.lr_func = lr_func
        self.loss_func = loss_func
        self.layers = layers

    def train(self, data, labels):
        pass

    def compute_loss(self):
        pass
