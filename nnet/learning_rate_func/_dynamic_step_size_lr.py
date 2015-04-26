from __future__ import absolute_import, division

from ._base import LearningRateFuncBase
import numpy as np

class DynamicStepSizeLR(LearningRateFuncBase):

    def __init__(self, eta_0, gamma, k, threshold_0):
        self.eta_0 = eta_0
        self.threshold_0 = threshold_0
        self.gamma = gamma
        self.k = k
        self.drop = 0
        self.last_update = None

    def apply(self, t, net):
        self._calc_drop(t, net)
        return self.eta_0 \
               * pow(self.gamma, self.drop)

    def _calc_drop(self, t, net):
        latest_update = len(net.losses)

        if self.last_update == latest_update:
            # Already checked for this update.
            pass
        elif latest_update < self.k:
            # Not enough data.
            pass
        else:
            self.last_update = latest_update

            mean_loss = np.mean(
                net.losses[latest_update-self.k:latest_update, 3]
            )

            # Increase drop if the last mean loss of
            # last k update is smaller than current threshold.
            if mean_loss < self.threshold_0 * pow(self.gamma, self.drop):
                self.drop += 1
