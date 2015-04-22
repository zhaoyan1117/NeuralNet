from __future__ import absolute_import, division

import numpy as np

def normalize(data):
    normalized = np.zeros(data.shape)
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    for i in xrange(len(data)):
        normalized[i] = (data[i] - means[i]) \
                        / stds[i]

    return normalized
