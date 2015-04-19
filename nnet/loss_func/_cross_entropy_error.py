from __future__ import absolute_import, division

import cudamat as cm

from ._base import LossFuncBase

class CrossEntropyError(LossFuncBase):

    def apply(self, y, y_hat):
        entropy = cm.empty(y_hat.shape)
        cm.log(y_hat, target=entropy)
        entropy.mult(y)

        n_y_hat = cm.empty(y_hat.shape)
        y_hat.mult(-1, n_y_hat)
        n_y_hat.add(1)

        cm.log(n_y_hat)

        n_y = cm.empty(y.shape)
        y.mult(-1, n_y)
        n_y.add(1)
        n_y_hat.mult(n_y)
        entropy.add(n_y_hat)

        entropy_sum = self.find_sum(entropy)
        size = entropy.shape[0]

        del entropy
        del n_y_hat
        del n_y

        return -entropy_sum / float(size)

    def apply_derivative(self, y, y_hat):
        diff = cm.empty(y.shape)
        y_hat.subtract(y, diff)

        n_y_hat = cm.empty(y_hat.shape)
        y_hat.mult(-1, n_y_hat)
        n_y_hat.add(1)
        n_y_hat.mult(y_hat)

        diff.divide(n_y_hat)
        diff.divide(float(diff.shape[0]))

        del n_y_hat

        return diff
