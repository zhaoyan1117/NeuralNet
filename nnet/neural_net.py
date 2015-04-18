from __future__ import absolute_import

import numpy as np

from .neural_net_exception import NeuralNetException
from .util import vectorize_labels, devectorize_labels, shuffle_data_labels

class NeuralNet(object):

    def __init__(self, batch_size, lr_func, loss_func,
                 stopping_c, layers, **kwargs):
        self.batch_size = batch_size
        self.lr_func = lr_func
        self.loss_func = loss_func
        self.stopping_c = stopping_c
        self.layers = layers

        self.data, self.labels = None, None

        self.loss_period = kwargs.get('loss_period', 200)
        self.check_period = kwargs.get('check_period')
        self.status_period = kwargs.get('status_period', 500)

    def train(self, data, labels):
        losses = np.empty((0, 2))

        # Shuffle input data and labels.
        self.data, self.labels = \
            shuffle_data_labels(data, vectorize_labels(labels))

        self.cur_epoch = 0
        data_i = 0
        while not self.stopping_c.stop(self):
            # Slice data and labels for this epoch.
            cur_data = self.data[data_i:data_i+self.batch_size]
            cur_label = self.labels[data_i:data_i+self.batch_size]

            # Forward propagation.
            y_hat = self._forward_p(cur_data)

            # Backward propagation.
            self._backward_p(self._compute_loss(cur_label, y_hat))

            # Gradient descent update.
            lr = self.lr_func(self.cur_epoch)
            self._update(lr)

            if not self.cur_epoch % self.loss_period:
                losses = np.append(losses,
                                   [[self.cur_epoch, self.compute_all_loss()]],
                                   axis=0)

            if not self.cur_epoch % self.check_period:
                self._numerical_check()

            if not self.cur_epoch % self.status_period:
                print "EPOCH: {epoch} | Score: {score}"\
                    .format(epoch=self.cur_epoch, score=self.score(self.data, self.labels))

            # Update cur_epoch and data index.
            self.cur_epoch += 1
            data_i += self.batch_size

            # After one full round on input data,
            # reshuffle data and labels and reset data index.
            if data_i >= len(self.data):
                data_i = 0
                self.data, self.labels = \
                    shuffle_data_labels(data, vectorize_labels(labels))

    def predict(self, data):
        vectorized = self._forward_p(data)
        return devectorize_labels(vectorized)

    def score(self, data, labels):
        predictions = self.predict(data)
        correct = predictions == labels
        return np.count_nonzero(correct) / float(len(labels))

    def _forward_p(self, data):
        cur_z = data
        for l in self.layers:
            cur_z = l.forward_p(cur_z)
        return cur_z

    def _backward_p(self, delta):
        cur_delta = delta
        for l in reversed(self.layers):
            cur_delta = l.backward_p(cur_delta)

    def _update(self, lr):
        for l in self.layers:
            l.update(lr)

    def _numerical_check(self):
        for l in self.layers:
            passed = l.numerical_check(self)
            if not passed:
                print "WARNING: layer {0} " \
                      "failed numerical check.".format(l.level)

    def _compute_loss(self, labels, predictions):
        return self.loss_func.apply(labels, predictions)

    def compute_all_loss(self):
        if self.data is None:
            raise NeuralNetException('Cannot compute loss without data.')

        return self._compute_loss(self.labels,
                                  self._forward_p(self.data))
