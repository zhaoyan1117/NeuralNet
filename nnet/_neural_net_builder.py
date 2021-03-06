from __future__ import absolute_import

import cPickle as pickle
import time

import cudamat as cm

import nnet.activation_func as af
import nnet.layer as layer
import nnet.learning_rate_func as lrf
import nnet.stopping_criteria as sc
from nnet._neural_net import NeuralNet
from nnet._neural_net_exception import NeuralNetException


class NeuralNetBuilder(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_level = 1
        self.layers = []
        self.batch_size = 10
        self.lr_func = None
        self.stopping_c = sc.OrConditions()
        self.status_period = 10000

    def load(self, file_name):
        cm.CUDAMatrix.init_random(int(time.time()))

        with open(file_name, 'r') as fd:
            net = pickle.load(fd)
            for l in net.layers:
                l.load_params()

        # Change to new learning func if given.
        if self.lr_func:
            net.lr_func = self.lr_func
        return net

    def build(self):
        if not self.layers:
            raise NeuralNetException('Not enough layers.')
        elif not isinstance(self.layers[-1], layer.OutputLayer):
            raise NeuralNetException('Last layer must be OutputLayer.')
        elif self.lr_func is None:
            raise NeuralNetException('No learning rate function.')
        elif self.stopping_c.is_empty:
            raise NeuralNetException('No stopping criteria.')
        else:
            for l in self.layers:
                l.init(self.batch_size)

            return NeuralNet(self.batch_size,
                             self.lr_func,
                             self.stopping_c,
                             self.layers,
                             self.status_period)

    def get_act_func(self, act_func):
        if act_func == 'Dummy':
            return af.Dummy()
        elif act_func == 'Sigmoid':
            return af.Sigmoid()
        elif act_func == 'Tanh':
            return af.Tanh()
        elif act_func == 'ReLU':
            return af.ReLU()
        else:
            raise NeuralNetException(
                'Activation function {0} does not exist.'.format(act_func)
            )

    def add_fully_connected_layer(self, size, act_func,
                                  sigma='c', use_bias=True, **kwargs):
        cur_layer = layer.FullyConnectedLayer(self.cur_level,
                                              size,
                                              self.get_act_func(act_func),
                                              sigma,
                                              use_bias,
                                              **kwargs)

        if self.layers:
            self.layers[-1].set_next_layer_size(cur_layer.size)

        self.layers.append(cur_layer)
        self.cur_level += 1
        return self

    def add_output_layer(self, size, act_func, loss_func):
        cur_layer = layer.OutputLayer(self.cur_level,
                                      size,
                                      self.get_act_func(act_func),
                                      loss_func)

        if self.layers:
            self.layers[-1].set_next_layer_size(cur_layer.size)

        self.layers.append(cur_layer)
        self.cur_level += 1
        return self

    def add_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def add_step_size_lr_func(self, eta_0, gamma, step_size):
        self.lr_func = lrf.StepSizeLR(eta_0, gamma, step_size)
        return self

    def add_dynamic_step_size_lr_func(self, eta_0, gamma, k, threshold_0):
        self.lr_func = lrf.DynamicStepSizeLR(eta_0, gamma, k, threshold_0)
        return self

    def add_inv_prop_lr_func(self, eta_0, lbd):
        self.lr_func = lrf.InvPropLR(eta_0, lbd)
        return self

    def add_constant_lr_func(self, eta_0):
        self.lr_func = lrf.ConstantLR(eta_0)
        return self

    def add_max_epoch_stopping_criteria(self, max_epoch):
        self.stopping_c.add_sc(sc.MaxEpoch(max_epoch))
        return self

    def add_min_improve_stopping_criteria(self, k, threshold):
        self.stopping_c.add_sc(sc.MinImproveScore(k, threshold))
        return self

    def add_status_period(self, status_period):
        self.status_period = status_period
        return self
