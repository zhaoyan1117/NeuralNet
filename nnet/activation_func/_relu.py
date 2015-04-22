from __future__ import absolute_import, division

from ._base import ActivationFuncBase


class ReLU(ActivationFuncBase):

    def apply(self, z, a):
        z.maximum(0.0, a)

    def apply_scalar(self, s):
        return max(0.0, s)

    def mult_with_derivative(self, target, z, a):
        z.greater_than(0.0, a)
        target.mult(a)
