from __future__ import absolute_import, division

import cudamat as cm

from ._base import ActivationFuncBase

class Sigmoid(ActivationFuncBase):

    def apply(self, z):
        z.apply_sigmoid()

    def mult_with_derivative(self, target, sig_z):
        cm.learn.mult_by_sigmoid_deriv(target, sig_z)
