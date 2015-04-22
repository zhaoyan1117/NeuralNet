from __future__ import absolute_import, division

import numpy as np

def normalize(data):
    def normalize_col(col):
        mean_col = np.mean(col)
        std_col = np.std(col)
        if std_col != 0.0:
            return (col - mean_col) / std_col
        else:
            return (col - mean_col)
    return np.apply_along_axis(normalize_col, 0, data)

def log_plus(data):
    return np.log(data + 0.1)

def binarize(data):
    return (data > 0).astype(data.dtype)
