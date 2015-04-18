from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class LayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_p(self, z):
        pass

    @abstractmethod
    def backward_p(self, next_delta):
        pass

    @abstractmethod
    def update(self, epoch):
        pass

    @abstractmethod
    def numerical_check(self, net):
        pass
