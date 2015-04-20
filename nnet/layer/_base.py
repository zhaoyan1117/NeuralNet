from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class LayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_next_layer_size(self, next_size):
        pass

    @abstractmethod
    def forward_p(self, z):
        pass

    @abstractmethod
    def backward_p(self, next_delta_or_y):
        pass

    @abstractmethod
    def update(self, epoch):
        pass

    def _free_mem(self):
        if hasattr(self, 'z'):
            del self.z
        if hasattr(self, 'a'):
            del self.a
        if hasattr(self, 'next_z'):
            del self.next_z
        if hasattr(self, 'my_delta'):
            del self.my_delta
