from __future__ import absolute_import, division

import cudamat as cm

from ._base import ActivationFuncBase


class Tanh(ActivationFuncBase):

    def apply(self, z):
        th = cm.empty(z.shape)
        z.apply_tanh(th)
        return th

    def apply_derivative(self, z):
        th = cm.empty(z.shape)
        z.apply_tanh(th)

        cm.pow(th, 2, th)
        th.mult(-1).add(1)
        return th
