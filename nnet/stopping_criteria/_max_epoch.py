from __future__ import absolute_import

from ._base import StoppingCriteriaBase

class MaxEpoch(StoppingCriteriaBase):

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def stop(self, net):
        return net.cur_epoch > self.max_epoch
