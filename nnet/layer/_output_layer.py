from __future__ import absolute_import, division

import cudamat as cm

from ._base import LayerBase

class OutputLayer(LayerBase):

    def __init__(self, level, size, activation_func):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.z = None
        self.a = None
        self.delta = None

    def set_next_layer_size(self, next_size):
        # Output layer does not have next layer.
        pass

    def forward_p(self, z):
        del self.z
        del self.a

        self.z = z
        self.a = self.activation_func.apply(self.z)
        return self.a

    def backward_p(self, next_delta):
        del self.delta

        self.delta = cm.empty(next_delta.shape)
        next_delta.mult(self.activation_func.apply_derivative(self.z),
                        self.delta)

        return self.delta

    def update(self, lr):
        # No weights to update for output layer.
        return None
