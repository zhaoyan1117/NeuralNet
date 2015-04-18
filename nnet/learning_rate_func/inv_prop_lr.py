from __future__ import absolute_import, division

from .base import LearningRateFuncBase

class InvPropLR(LearningRateFuncBase):

    def __init__(self, init_rate, period):
        self.init_rate = init_rate
        self.period = period

    def apply(self, epoch):
        return self.init_rate / \
               (epoch / self.period)
