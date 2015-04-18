from __future__ import absolute_import, division

import numpy as np

from .loss_func_base import LossFuncBase

class CrossEntropyError(LossFuncBase):

    def apply(self, y, y_hat):
        entropy = y * np.log(y_hat) \
                  + (1.0 - y) * np.log(1.0 - y_hat)
        return -np.sum(entropy) \
               / float(len(entropy))

    def apply_derivative(self, y, y_hat):
        diff = y_hat - y
        return diff \
               / (y_hat * (1.0 - y_hat)) \
               / float(len(diff))
