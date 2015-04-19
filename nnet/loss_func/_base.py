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

    def find_sum(self, mat):
        col_sum = mat.sum(axis=0)
        row_sum = col_sum.sum(axis=1)
        sum = row_sum.asarray()[0][0]
        del col_sum
        del row_sum
        return sum
