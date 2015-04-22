from __future__ import absolute_import, division

import math

import cudamat as cm

from ._base import ActivationFuncBase


class Tanh(ActivationFuncBase):

    def apply(self, z, a):
        z.apply_tanh(a)

    def apply_scalar(self, s):
        return math.tanh(s)

    def mult_with_derivative(self, target, z, a):
        cm.pow(a, 2, a)
        a.mult(-1).add(1)
        target.mult(a)
