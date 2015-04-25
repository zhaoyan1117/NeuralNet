from __future__ import absolute_import

import time
import cPickle as pickle

import numpy as np
import cudamat as cm

from .util import vectorize_labels, devectorize_labels, shuffle_data_labels


class NeuralNet(object):

    def __init__(self, batch_size, lr_func, stopping_c, layers,
                 status_period=10000, **kwargs):
        cm.CUDAMatrix.init_random(int(time.time()))
        # TODO: Currently only support batch size divide data size.
        self.batch_size = batch_size
        self.lr_func = lr_func
        self.stopping_c = stopping_c
        self.layers = layers
        self.status_period = status_period

    def train(self, data, labels):
        start = time.time()

        self.losses = np.empty((0, 5))

        self.klasses = np.unique(labels)

        # Shuffle input data and labels.
        self.shuffled_data, self.shuffled_labels = \
            shuffle_data_labels(data, labels)
        self.shuffled_vec_labels = \
            vectorize_labels(self.shuffled_labels, len(self.klasses))
        self.cu_data, self.cu_vec_labels = \
            cm.CUDAMatrix(self.shuffled_data), cm.CUDAMatrix(self.shuffled_vec_labels)

        self.cur_data = cm.empty((self.batch_size, self.cu_data.shape[1]))
        self.cur_vec_labels = cm.empty((self.batch_size, self.cu_vec_labels.shape[1]))

        data_i = 0
        self.cur_epoch = 0
        self.cur_iteration = 0

        while not self.stopping_c.stop(self):
            # Slice data and labels for this epoch.
            self.cu_data.get_row_slice(data_i, data_i+self.batch_size, self.cur_data)
            self.cu_vec_labels.get_row_slice(data_i, data_i+self.batch_size, self.cur_vec_labels)

            # Forward propagation.
            self._forward_p(
                self.cur_data
            )

            # Backward propagation.
            self._backward_p(
                self.cur_vec_labels
            )

            # Gradient descent update.
            self._update(
                self.lr_func.apply(self.cur_iteration)
            )

            # Do periodic job.
            if not self.cur_iteration % self.status_period:
                time_elapsed = time.time() - start

                score, loss = self._compute_training_score_n_loss()

                print "Epoch: {:3d} | " \
                      "Iteration: {:3d} x {status_period} | " \
                      "Score: {:13.12f} | " \
                      "Loss: {:13.10f} | " \
                      "Time elapsed: {:3.2f} minutes" \
                    .format(self.cur_epoch,
                            self.cur_iteration / self.status_period,
                            score, loss, time_elapsed/60,
                            status_period=self.status_period)

                self.losses = np.append(self.losses,
                                        [[self.cur_epoch, self.cur_iteration, score, loss, time_elapsed]],
                                        axis=0)

            # Update cur_iteration and data index.
            data_i += self.batch_size
            self.cur_iteration += 1

            # Finished one epoch.
            if data_i + self.batch_size > len(self.shuffled_data):
                data_i = 0
                self.cur_epoch += 1

                # Re-shuffle data.
                self.cu_data.free_device_memory()
                self.cu_vec_labels.free_device_memory()
                del self.cu_data, self.cu_vec_labels
                self.shuffled_data, self.shuffled_labels = \
                    shuffle_data_labels(data, labels)
                self.shuffled_vec_labels = \
                    vectorize_labels(self.shuffled_labels, len(self.klasses))
                self.cu_data, self.cu_vec_labels = \
                    cm.CUDAMatrix(self.shuffled_data), cm.CUDAMatrix(self.shuffled_vec_labels)

        # Free memory.
        self.cur_data.free_device_memory()
        self.cur_vec_labels.free_device_memory()
        del self.cur_data, self.cur_vec_labels

        self.cu_data.free_device_memory()
        self.cu_vec_labels.free_device_memory()
        del self.cu_data, self.cu_vec_labels

        del self.shuffled_data, self.shuffled_labels
        del self.shuffled_vec_labels

        duration = (time.time() - start) / 60
        print "Training takes {0} minutes.".format(duration)

        # Return losses.
        return self.losses

    def predict(self, data):
        predicted = np.empty((0,))

        cu_data = cm.CUDAMatrix(data)
        cur_data = cm.empty((self.batch_size, data.shape[1]))

        for data_i in xrange(0, len(data), self.batch_size):
            cu_data.get_row_slice(data_i, data_i+self.batch_size, cur_data)

            predicted = \
                np.append(predicted, devectorize_labels(self._forward_p(cur_data, True).asarray()))

        # Free memory.
        cu_data.free_device_memory()
        cur_data.free_device_memory()
        del cu_data, cur_data
        return predicted

    def score(self, data, labels):
        predictions = self.predict(data)
        correct = predictions == labels
        return np.count_nonzero(correct) / float(len(labels))

    def _compute_training_score_n_loss(self):
        predictions = np.empty((0,))
        loss = 0.0
        for data_i in xrange(0, len(self.shuffled_data), self.batch_size):
            self.cu_data.get_row_slice(data_i, data_i+self.batch_size, self.cur_data)
            self.cu_vec_labels.get_row_slice(data_i, data_i+self.batch_size, self.cur_vec_labels)


            predictions = \
                np.append(predictions,
                          devectorize_labels(self._forward_p(self.cur_data, True).asarray()))
            loss += self.output_layer.compute_loss(self.cur_vec_labels) * float(self.batch_size)

        correct = predictions == self.shuffled_labels
        score = np.count_nonzero(correct) / float(len(predictions))
        loss /= len(self.shuffled_data)
        return score, loss

    def _forward_p(self, data, predict=False):
        cur_z = data
        for l in self.layers:
            cur_z = l.forward_p(cur_z, predict)
        return cur_z

    def _backward_p(self, y):
        delta_or_y = y
        for l in reversed(self.layers):
            delta_or_y = l.backward_p(delta_or_y)

    def _update(self, lr):
        for l in self.layers:
            l.update(lr)

    @property
    def output_layer(self):
        return self.layers[-1]

    def dump(self, file_name):
        for l in self.layers:
            l.dump_params()

        with open(file_name, 'w') as fd:
            pickle.dump(self, fd)
