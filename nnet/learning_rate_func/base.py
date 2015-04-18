from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class LearningRateFuncBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, epoch):
        pass
