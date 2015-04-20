from __future__ import absolute_import, division

from ._base import LearningRateFuncBase

class StepSizeLR(LearningRateFuncBase):

    def __init__(self, eta_0, gamma, step_size):
        self.eta_0 = eta_0
        self.gamma = gamma
        self.step_size = step_size

    def apply(self, t):
        return self.eta_0 \
               * pow(self.gamma, t // self.step_size)
