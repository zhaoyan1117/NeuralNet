from __future__ import absolute_import, division

from ._base import ActivationFuncBase


class ReLU(ActivationFuncBase):

    def apply(self, z):
        z.maximum(0.0)

    def apply_scalar(self, s):
        return max(0.0, s)

    def mult_with_derivative(self, target, activated_z):
        activated_z.greater_than(0.0)
        target.mult(activated_z)
