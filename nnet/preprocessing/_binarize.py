from __future__ import absolute_import, division

from ._base import PreprocessingBase


class Binarize(PreprocessingBase):

    def fit(self, X):
        return (X > 0).astype(X.dtype)

    def transform(self, X):
        return (X > 0).astype(X.dtype)
