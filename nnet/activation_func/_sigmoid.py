from __future__ import absolute_import, division

import math

import cudamat as cm

from ._base import ActivationFuncBase


class Sigmoid(ActivationFuncBase):

    def apply(self, z, a):
        z.apply_sigmoid(a)

    def apply_scalar(self, s):
        return 1.0 \
               / (1.0 + math.exp(-s))

    def mult_with_derivative(self, target, z, a):
        cm.learn.mult_by_sigmoid_deriv(target, a)
