from __future__ import absolute_import

import numpy as np

def vectorize_labels(labels, vec_len):
    vectorized = np.zeros((len(labels), vec_len))
    for i in xrange(len(labels)):
        vectorized[i][labels[i]] = 1
    return vectorized

def devectorize_labels(vectorized):
    devectorized = np.zeros((len(vectorized),))
    for i in xrange(len(vectorized)):
        devectorized[i] = np.argmax(vectorized[i])
    return devectorized
