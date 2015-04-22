from __future__ import absolute_import, division

import math

import cudamat as cm

from ._base import ActivationFuncBase


class Softplus(ActivationFuncBase):

    def apply(self, z, a):
        cm.exp(z, a)
        a.add(1)
        cm.log(a, a)

    def apply_scalar(self, s):
        return math.log(1 + math.exp(s))

    def mult_with_derivative(self, target, z, a):
        z.apply_sigmoid(a)
        target.mult(a)
