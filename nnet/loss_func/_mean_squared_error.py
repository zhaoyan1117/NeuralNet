from __future__ import absolute_import, division

import cudamat as cm

from ._base import LossFuncBase

class MeanSquaredError(LossFuncBase):

    def apply(self, y, y_hat):
        # Diff.
        diff = cm.empty(y.shape)
        y.subtract(y_hat, diff)

        # Square.
        cm.pow(diff, 2, diff)

        diff_sum = self.find_sum(diff)
        size = diff.shape[0]
        del diff

        return diff_sum / float(2*size)

    def apply_derivative(self, y, y_hat):
        diff = cm.empty(y.shape)
        y_hat.subtract(y, diff)

        return diff.divide(float(diff.shape[0]))
