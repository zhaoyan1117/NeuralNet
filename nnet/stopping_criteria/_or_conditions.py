from __future__ import absolute_import

from ._base import StoppingCriteriaBase


class OrConditions(StoppingCriteriaBase):

    def __init__(self):
        self.scs = []

    def add_sc(self, sc):
        self.scs.append(sc)

    @property
    def is_empty(self):
        return len(self.scs) == 0

    def stop(self, net):
        for sc in self.scs:
            if sc.stop(net):
                return True
