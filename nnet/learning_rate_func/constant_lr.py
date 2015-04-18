from __future__ import absolute_import, division

from .base import LearningRateFuncBase

class ConstantLR(LearningRateFuncBase):

    def __init__(self, init_rate):
        self.init_rate = init_rate

    def apply(self, epoch):
        return self.init_rate
