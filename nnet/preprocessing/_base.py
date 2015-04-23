from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class PreprocessingBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass
