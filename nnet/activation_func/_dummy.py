from __future__ import absolute_import, division

from ._base import ActivationFuncBase

class Dummy(ActivationFuncBase):
    """
    For input layer.
    """

    def apply(self, z):
        pass

    def apply_scalar(self, s):
        return s

    def mult_with_derivative(self, target, activated_z):
        pass
