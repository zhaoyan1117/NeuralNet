from __future__ import absolute_import, division

import numpy as np

from ._base import ActivationFuncBase

class Sigmoid(ActivationFuncBase):

    def apply(self, z):
        return 1.0 \
               / (1.0 + np.exp(-z))

    def apply_derivative(self, z):
        exp_z = np.exp(z)
        return exp_z \
               / ((1.0 + exp_z) ** 2)
