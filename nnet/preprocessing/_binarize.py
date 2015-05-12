from __future__ import absolute_import, division

from ._base import PreprocessingBase


class Binarize(PreprocessingBase):

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X):
        return (X > self.threshold).astype(X.dtype)

    def transform(self, X):
        return (X > self.threshold).astype(X.dtype)
