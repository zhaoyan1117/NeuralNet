from __future__ import absolute_import, division

import numpy as np

from ._base import PreprocessingBase


class Normalize(PreprocessingBase):

    def fit(self, X):
        self.means = np.empty((0,))
        self.stds = np.empty((0,))

        transformed = np.empty(X.shape)
        for c in xrange(X.shape[1]):
            transformed[:,c] = X[:,c]

            mean_col = np.mean(transformed[:,c])
            std_col = np.std(transformed[:,c])
            self.means = np.append(self.means, mean_col)
            self.stds = np.append(self.stds, std_col)

            transformed[:,c] -= mean_col
            if std_col != 0.0:
                transformed[:,c] /= std_col
        return transformed

    def transform(self, X):
        transformed = np.empty(X.shape)
        for c in xrange(X.shape[1]):
            transformed[:,c] = X[:,c]
            mean_col = self.means[c]
            std_col = self.stds[c]
            transformed[:,c] -= mean_col
            if std_col != 0.0:
                transformed[:,c] /= std_col
        return transformed
