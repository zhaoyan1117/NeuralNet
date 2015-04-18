from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class LossFuncBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, y, y_hat):
        pass

    @abstractmethod
    def apply_derivative(self, y, y_hat):
        pass
