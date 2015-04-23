from __future__ import absolute_import, division

import numpy as np

from ._base import PreprocessingBase


class LogPlus(PreprocessingBase):

    def fit(self, X):
        return np.log(X + 0.1)

    def transform(self, X):
        return np.log(X + 0.1)
