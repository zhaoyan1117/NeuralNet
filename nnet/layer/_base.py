from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

import cudamat as cm

class LayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_next_layer_size(self, next_size):
        pass

    @abstractmethod
    def init(self, batch_size):
        pass

    @abstractmethod
    def forward_p(self, z, predict=False):
        pass

    @abstractmethod
    def backward_p(self, next_delta_or_y):
        pass

    @abstractmethod
    def forward_p_single(self, single_z):
        pass

    @abstractmethod
    def update(self, epoch):
        pass

    @abstractmethod
    def dump_params(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

    def _dump_np(self, name):
        np_name = 'np_' + name
        self.__dict__[np_name] = self.__dict__[name].asarray()
        self.__dict__[name].free_device_memory()
        del self.__dict__[name]

    def _load_np(self, name):
        np_name = 'np_' + name
        self.__dict__[name] = \
            cm.CUDAMatrix(self.__dict__[np_name])
        del self.__dict__[np_name]
