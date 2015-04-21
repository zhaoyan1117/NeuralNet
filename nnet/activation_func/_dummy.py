from __future__ import absolute_import, division

from ._base import ActivationFuncBase

class Dummy(ActivationFuncBase):
    """
    For input layer.
    """

    def apply(self, z):
        pass

    def mult_with_derivative(self, target, dummy_z):
        pass