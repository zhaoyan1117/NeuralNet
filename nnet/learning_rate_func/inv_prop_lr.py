from __future__ import absolute_import, division

from .base import LearningRateFuncBase

class InvPropLR(LearningRateFuncBase):

    def __init__(self, eta_0, lbd):
        self.eta_0 = eta_0
        self.lbd = lbd

    def apply(self, epoch):
        return self.eta_0 \
               / (1 + self.eta_0 * self.lbd * epoch)
