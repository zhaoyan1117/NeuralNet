from __future__ import absolute_import

import nnet.activation_func as af
import nnet.layer as layer
import nnet.learning_rate_func as lrf
import nnet.loss_func as lf
import nnet.stopping_criteria as sc
from nnet.neural_net import NeuralNet
from nnet.neural_net_exception import NeuralNetException

class NeuralNetBuilder(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_level = 1
        self.layers = []
        self.lr_func = None
        self.loss_func = None
        self.stopping_c = None

        self.batch_size = 10
        self.loss_period = 200
        self.check_period = None
        self.status_period = 500

    def build(self):
        if not self.layers:
            raise NeuralNetException('Not enough layers.')
        elif not isinstance(self.layers[-1], layer.OutputLayer):
            raise NeuralNetException('Last layer must be OutputLayer.')
        elif self.lr_func is None:
            raise NeuralNetException('No learning rate function.')
        elif self.loss_func is None:
            raise NeuralNetException('No loss function.')
        elif self.stopping_c is None:
            raise NeuralNetException('No stopping criteria.')
        else:
            return NeuralNet(self.batch_size,
                             self.lr_func,
                             self.loss_func,
                             self.stopping_c,
                             self.layers,
                             loss_period=self.loss_period,
                             check_period=self.check_period,
                             status_period=self.status_period)

    def get_act_func(self, act_func):
        if act_func == 'Dummy':
            return af.Dummy()
        elif act_func == 'Sigmoid':
            return af.Sigmoid()
        elif act_func == 'Tanh':
            return af.Tanh()
        else:
            raise NeuralNetException(
                'Activation function {0} does not exist.'.format(act_func)
            )

    def add_fully_connected_layer(self, size, act_func, sigma=1.0, bias=True):
        if self.layers:
            self.layers[-1].set_next_layer_size(size)

        self.layers.append(
            layer.FullyConnectedLayer(self.cur_level,
                                      size,
                                      self.get_act_func(act_func),
                                      sigma,
                                      bias)
        )

        return self
