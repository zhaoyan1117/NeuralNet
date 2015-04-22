from __future__ import absolute_import, division

import math

import cudamat as cm

from ._base import ActivationFuncBase


class Tanh(ActivationFuncBase):

    def apply(self, z):
        z.apply_tanh()

    def apply_scalar(self, s):
        return math.tanh(s)

    def mult_with_derivative(self, target, activated_z):
        cm.pow(activated_z, 2, activated_z)
        activated_z.mult(-1).add(1)
        target.mult(activated_z)
