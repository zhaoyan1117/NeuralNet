from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class ActivationFuncBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, z):
        pass

    @abstractmethod
    def apply_scalar(self, s):
        pass

    @abstractmethod
    def mult_with_derivative(self, target, applied_z):
        pass
