from __future__ import absolute_import

import numpy as np
import cudamat as cm

from ._neural_net_exception import NeuralNetException
from .util import vectorize_labels, devectorize_labels, shuffle_data_labels

class NeuralNet(object):

    def __init__(self, batch_size, lr_func, loss_func,
                 stopping_c, layers, **kwargs):
        self.batch_size = batch_size
        self.lr_func = lr_func
        self.loss_func = loss_func
        self.stopping_c = stopping_c
        self.layers = layers

        self.data, self.vec_labels = None, None

        self.status_period = kwargs.get('status_period', 10000)

    def train(self, data, labels):
        self.klasses = np.unique(labels)
        losses = np.empty((0, 3))

        # Shuffle input data and labels.
        self.data, self.vec_labels = \
            shuffle_data_labels(data, vectorize_labels(labels, len(self.klasses)))
        self.cu_data = cm.CUDAMatrix(self.data)
        self.cu_vec_labels = cm.CUDAMatrix(self.vec_labels)

        cur_data = cm.empty((self.batch_size, self.cu_data.shape[1]))
        cur_labels = cm.empty((self.batch_size, self.cu_vec_labels.shape[1]))

        data_i = 0
        self.cur_epoch = 0
        self.cur_iteration = 0

        while not self.stopping_c.stop(self):
            # Slice data and labels for this epoch.
            self.cu_data.get_row_slice(data_i, data_i+self.batch_size, cur_data)
            self.cu_vec_labels.get_row_slice(data_i, data_i+self.batch_size, cur_labels)

            # Forward propagation.
            y_hat = self._forward_p(cur_data)

            # Backward propagation.
            self._backward_p(
                self.loss_func.apply_derivative(cur_labels, y_hat)
            )

            # Gradient descent update.
            self._update(
                self.lr_func.apply(self.cur_iteration)
            )

            # Do periodic job.
            if not self.cur_iteration % self.status_period:
                print "EPOCH: {epoch} | Score: {score}"\
                    .format(epoch=self.cur_epoch, score=self.score(self.cu_data, labels))

            # Update cur_iteration and data index.
            data_i += self.batch_size
            self.cur_iteration += 1
            del y_hat

            # Finished one epoch.
            if data_i >= len(self.data):
                losses = np.append(losses,
                                   [[self.cur_epoch,
                                     self.compute_all_loss(),
                                     self.score(self.cu_data, labels)]],
                                   axis=0)

                # Shuffle data.
                self.data, self.vec_labels = \
                    shuffle_data_labels(data, vectorize_labels(labels, len(self.klasses)))
                del self.cu_data
                del self.cu_vec_labels
                self.cu_data = cm.CUDAMatrix(self.data)
                self.cu_vec_labels = cm.CUDAMatrix(self.vec_labels)

                data_i = 0
                self.cur_epoch += 1

    def predict(self, data):
        if isinstance(data, np.ndarray):
            is_local = True
            cu_data = cm.CUDAMatrix(data)
        else:
            is_local = False
            cu_data = data

        cu_vectorized = self._forward_p(cu_data)
        vectorized = cu_vectorized.asarray()

        del cu_vectorized
        if is_local:
            del cu_data
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

    def _compute_loss(self, vec_labels, predictions):
        return self.loss_func.apply(vec_labels, predictions)

    def compute_all_loss(self):
        if self.data is None:
            raise NeuralNetException('Cannot compute loss without data.')

        return self._compute_loss(self.cu_vec_labels,
                                  self._forward_p(self.cu_data))
