from __future__ import absolute_import

import numpy as np

from ._base import StoppingCriteriaBase


class MinImproveScore(StoppingCriteriaBase):

    def __init__(self, k, threshold):
        self.k = k
        self.threshold = threshold
        self.last_update = None

    def stop(self, net):
        latest_update = len(net.losses)

        if self.last_update == latest_update:
            # Already checked for this update.
            return False
        elif latest_update < 2 * self.k:
            # Not enough data.
            return False
        else:
            self.last_update = latest_update

            cur_loss = np.mean(
                net.losses[latest_update-self.k:latest_update, 3]
            )

            last_loss = np.mean(
                net.losses[latest_update-2*self.k:latest_update-self.k, 3]
            )

            if last_loss - cur_loss > self.threshold:
                return False
            else:
                print('Stop at epoch {epoch}, '
                      'iteration {iteration}\n'
                      'average loss of last {k} epochs '
                      'improved smaller than {threshold}\n'
                      'Latest score : {latest_score} | '
                      'Latest loss  : {latest_loss}'
                      .format(epoch=net.cur_epoch,
                              iteration=net.cur_iteration,
                              k=self.k,
                              threshold=self.threshold,
                              latest_score=net.losses[-1,2],
                              latest_loss=net.losses[-1,3]))
                return True
