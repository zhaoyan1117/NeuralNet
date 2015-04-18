from __future__ import absolute_import, division

import numpy as np

from .loss_func_base import LossFuncBase

class MeanSquaredError(LossFuncBase):

    def apply(self, y, y_hat):
        diff = y - y_hat
        return np.sum(diff**2) \
               / float(2*len(diff))

    def apply_derivative(self, y, y_hat):
        diff = y_hat - y
        return diff \
               / float(len(diff))
