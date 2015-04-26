from __future__ import absolute_import, division

from ._base import LearningRateFuncBase

class ConstantLR(LearningRateFuncBase):

    def __init__(self, eta_0):
        self.eta_0 = eta_0

    def apply(self, t, net):
        return self.eta_0
