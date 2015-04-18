from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class StoppingCriteriaBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def stop(self, net):
        pass
