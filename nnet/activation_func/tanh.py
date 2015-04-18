from __future__ import absolute_import, division

import numpy as np

from .base import ActivationFuncBase


class Tanh(ActivationFuncBase):

    def apply(self, z):
        return np.tanh(z)

    def apply_derivative(self, z):
        return 1.0 - (np.tanh(z) ** 2)
