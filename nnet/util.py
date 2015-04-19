from __future__ import absolute_import, division

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

def shuffle_data_labels(data, labels):
    assert len(data) == len(labels)
    indices = np.random.permutation(len(data))
    return data[indices], labels[indices]

def normalize(data):
    normalized = np.zeros(data.shape)
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    for i in xrange(len(data)):
        normalized[i] = (data[i] - means[i]) \
                        / stds[i]

    return normalized
