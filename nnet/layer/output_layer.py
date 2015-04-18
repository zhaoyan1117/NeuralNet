from __future__ import absolute_import, division

from .base import LayerBase

class OutputLayer(LayerBase):

    def __init__(self, size, activation_func, level):
        self.level = level
        self.size = size
        self.activation_func = activation_func
        self.z = None
        self.delta = None

    def forward_p(self, z):
        self.z = z
        return self.activation_func.apply(z)

    def backward_p(self, next_delta):
        self.delta = next_delta \
                     * self.activation_func.apply_derivative(self.z)
        return self.delta

    def update(self, lr):
        # No weights to update for output layer.
        return None
