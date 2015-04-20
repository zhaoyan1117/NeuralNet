from __future__ import absolute_import

import numpy as np

from ._base import StoppingCriteriaBase


class MinImproveScore(StoppingCriteriaBase):

    def __init__(self, k, threshold):
        self.k = k
        self.threshold = threshold
        self.last_epoch = None
        self.counter = 0

    def stop(self, net):
        latest_epoch = net.cur_epoch

        if self.last_epoch == latest_epoch:
            # Already checked for this epoch.
            return False
        elif latest_epoch < 2 * self.k:
            # Not enough data.
            return False
        else:
            self.last_epoch = latest_epoch

            cur_score = np.mean(
                net.losses[latest_epoch-self.k:latest_epoch, 1]
            )

            last_score = np.mean(
                net.losses[latest_epoch-2*self.k:latest_epoch-self.k, 1]
            )

            if cur_score - last_score > self.threshold:
                return False
            else:
                print('Stop at EPOCH {epoch}, '
                      'average score of last {k} epochs '
                      'improved smaller than {threshold}\n'
                      'Latest score : {latest_score} | '
                      'Latest loss  : {latest_loss}'
                      .format(epoch = latest_epoch,
                              k = self.k,
                              threshold = self.threshold,
                              latest_score = net.losses[-1,1],
                              latest_loss = net.losses[-1,2]))
                return True
