from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class LayerBase(object):
    __metaclass__ = ABCMeta

    @absolute_import
    def forward(self, z):
        pass
