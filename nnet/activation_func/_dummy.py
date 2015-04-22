from __future__ import absolute_import, division

from ._base import ActivationFuncBase

class Dummy(ActivationFuncBase):
    """
    For input layer.
    """

    def apply(self, z, a):
        z.mult(1.0, a)

    def apply_scalar(self, s):
        return s

    def mult_with_derivative(self, target, z, a):
        pass
