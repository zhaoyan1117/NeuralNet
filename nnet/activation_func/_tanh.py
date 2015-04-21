from __future__ import absolute_import, division

import cudamat as cm

from ._base import ActivationFuncBase

class Tanh(ActivationFuncBase):

    def apply(self, z):
        z.apply_tanh()

    def mult_with_derivative(self, target, tanh_z):
        cm.pow(tanh_z, 2, tanh_z)
        tanh_z.mult(-1).add(1)
        target.mult(tanh_z)
