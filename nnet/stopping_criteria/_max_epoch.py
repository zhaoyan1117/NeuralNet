from __future__ import absolute_import

from ._base import StoppingCriteriaBase

class MaxEpoch(StoppingCriteriaBase):

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def stop(self, net):
        if net.cur_epoch >= self.max_epoch:
            print('Stop at EPOCH {epoch}, '
                  'exceeds max epoch {max_epoch}'
                  .format(epoch=net.cur_epoch, max_epoch=self.max_epoch))
            return True
        else:
            return False
