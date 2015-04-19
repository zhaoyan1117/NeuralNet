from __future__ import absolute_import, division

from ._base import LearningRateFuncBase

class InvPropLR(LearningRateFuncBase):

    def __init__(self, eta_0, lbd):
        self.eta_0 = eta_0
        self.lbd = lbd

    def apply(self, t):
        return self.eta_0 \
               / (1 + pow(t, self.lbd))
