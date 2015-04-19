from __future__ import absolute_import, division

import cudamat as cm

from ._base import ActivationFuncBase

class Sigmoid(ActivationFuncBase):

    def apply(self, z):
        sig = cm.empty(z.shape)
        z.apply_sigmoid(sig)
        return sig

    def apply_derivative(self, z):
        sig = cm.empty(z.shape)
        z.apply_sigmoid(sig)

        one_m_sig = cm.empty(sig.shape)
        sig.mult(-1, one_m_sig)
        one_m_sig.add(1)
        sig.mult(one_m_sig)

        return sig
