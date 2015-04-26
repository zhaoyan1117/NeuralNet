from __future__ import absolute_import, division

import numpy as np

from ._base import PreprocessingBase

class NormalizeRmZeroStd(PreprocessingBase):
    """
    Not exact a PCA, only remove features with std 0.
    """

    def fit(self, X):
        self.means = np.empty((0,))
        self.stds = np.empty((0,))

        transformed = np.empty(X.shape)
        transformed_i = 0
        for c in xrange(X.shape[1]):
            col = X[:,c]
            mean_col = np.mean(col)
            std_col = np.std(col)
            self.means = np.append(self.means, mean_col)
            self.stds = np.append(self.stds, std_col)

            if std_col != 0.0:
                col -= mean_col
                col /= std_col
                transformed[:, transformed_i] = col
                transformed_i += 1

        self.num_features = transformed_i

        return transformed[:,:transformed_i]

    def transform(self, X):
        if len(X) == 0:
            # TODO: handle edge case nicely.
            return np.zeros((len(X), np.count_nonzero(self.stds)))

        transformed = np.empty(X.shape)
        transformed_i = 0
        non_zero_counter = 0
        for c in xrange(X.shape[1]):
            col = X[:,c]
            mean_col = self.means[c]
            std_col = self.stds[c]
            x_std_col = np.std(col)

            if x_std_col != 0 and std_col == 0:
                non_zero_counter += 1

            if std_col != 0.0:
                col -= mean_col
                col /= std_col
                transformed[:, transformed_i] = col
                transformed_i += 1

        if non_zero_counter != 0:
            print '{0} features with non zero std ' \
                  'are removed during this transform.'.format(non_zero_counter)

        assert transformed_i == np.count_nonzero(self.stds)
        return transformed[:,:transformed_i]
