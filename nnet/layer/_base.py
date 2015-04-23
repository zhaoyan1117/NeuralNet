from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

class LayerBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_next_layer_size(self, next_size):
        pass

    @abstractmethod
    def init_weights(self, batch_size):
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

    @abstractmethod
    def predict(self, z):
        pass

    def prediction_clean(self):
        if hasattr(self, 'predict_z'):
            self.predict_z.free_device_memory()
            del self.predict_z
