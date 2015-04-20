from __future__ import absolute_import

import time

import numpy as np
import cudamat as cm

from .util import vectorize_labels, devectorize_labels, shuffle_data_labels


class NeuralNet(object):

    def __init__(self, batch_size, lr_func, stopping_c, layers, **kwargs):
        self.batch_size = batch_size
        self.lr_func = lr_func
        self.stopping_c = stopping_c
        self.layers = layers

        self.print_period = kwargs.get('print_period',
                                       10000)
        self.status_period = kwargs.get('status_period',
                                        self.print_period)

    def train(self, data, labels):
        start = time.time()

        self.losses = np.empty((0, 4))

        self.klasses = np.unique(labels)
        vec_labels = vectorize_labels(labels, len(self.klasses))

        cu_no_shuffle_data = cm.CUDAMatrix(data)
        cu_no_shuffle_labels = cm.CUDAMatrix(vec_labels)

        # Shuffle input data and labels.
        self.data, self.labels = \
            shuffle_data_labels(data, vec_labels)
        self.cu_data, self.cu_labels = \
            cm.CUDAMatrix(self.data), cm.CUDAMatrix(self.labels)

        cur_data = cm.empty((self.batch_size, self.cu_data.shape[1]))
        cur_labels = cm.empty((self.batch_size, self.cu_labels.shape[1]))

        data_i = 0
        self.cur_epoch = 0
        self.cur_iteration = 0

        while not self.stopping_c.stop(self):
            # Slice data and labels for this epoch.
            self.cu_data.get_row_slice(data_i, data_i+self.batch_size, cur_data)
            self.cu_labels.get_row_slice(data_i, data_i+self.batch_size, cur_labels)

            # Forward propagation.
            self._forward_p(
                cur_data
            )

            # Backward propagation.
            self._backward_p(
                cur_labels
            )

            # Gradient descent update.
            self._update(
                self.lr_func.apply(self.cur_iteration)
            )

            # Do periodic job.
            if not self.cur_iteration % self.status_period\
                    or not self.cur_iteration % self.print_period:

                score, loss = self._training_score_n_loss(cu_no_shuffle_data,
                                                          cu_no_shuffle_labels,
                                                          labels)

                if not self.cur_iteration % self.print_period:
                    print "Epoch: {:4d} | " \
                          "Iteration: {:4d} x {print_period} | " \
                          "Score: {:13.12f} | " \
                          "Loss: {:13.10f}"\
                        .format(self.cur_epoch,
                                self.cur_iteration / self.print_period,
                                score, loss,
                                print_period=self.print_period)

                if not self.cur_iteration % self.status_period:
                    self.losses = np.append(self.losses,
                                            [[self.cur_epoch, self.cur_iteration, score, loss]],
                                            axis=0)

            # Update cur_iteration and data index.
            data_i += self.batch_size
            self.cur_iteration += 1

            # Finished one epoch.
            if data_i >= len(self.data):
                data_i = 0
                self.cur_epoch += 1

                # Re-shuffle data.
                del self.cu_data, self.cu_labels
                self.data, self.labels = \
                    shuffle_data_labels(data, vec_labels)
                self.cu_data, self.cu_labels = \
                    cm.CUDAMatrix(self.data), cm.CUDAMatrix(self.labels)

        # Free memory.
        del cur_data, cur_labels
        del self.data, self.labels
        del self.cu_data, self.cu_labels
        del cu_no_shuffle_data, cu_no_shuffle_labels

        duration = (time.time() - start) / 60
        print "Training takes {0} minutes.".format(duration)

        # Return losses.
        return self.losses

    def predict(self, data):
        cu_data = cm.CUDAMatrix(data)
        cu_predicted = self._forward_p(cu_data)
        vec_predicted = cu_predicted.asarray()
        del cu_predicted, cu_data
        return devectorize_labels(vec_predicted)

    def score(self, data, labels):
        predictions = self.predict(data)
        correct = predictions == labels
        return np.count_nonzero(correct) / float(len(labels))

    def _training_score_n_loss(self, cu_data, cu_labels, labels):
        loss, cu_predicted = self._compute_loss(cu_data, cu_labels)
        predictions = devectorize_labels(cu_predicted.asarray())
        del cu_predicted
        correct = predictions == labels
        score = np.count_nonzero(correct) / float(len(labels))
        return score, loss

    def _forward_p(self, data):
        cur_z = data
        for l in self.layers:
            cur_z = l.forward_p(cur_z)
        return cur_z

    def _backward_p(self, y):
        delta_or_y = y
        for l in reversed(self.layers):
            delta_or_y = l.backward_p(delta_or_y)

    def _update(self, lr):
        for l in self.layers:
            l.update(lr)

    def _compute_loss(self, cu_data, cu_labels):
        cu_predicted = self._forward_p(cu_data)
        return (
            self.layers[-1].compute_loss(cu_labels),
            cu_predicted
        )
