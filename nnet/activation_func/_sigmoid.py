from __future__ import absolute_import, division

import cudamat as cm

from ._base import ActivationFuncBase

class Sigmoid(ActivationFuncBase):

    def apply(self, z):
        self._free_memory()

        self._sig = cm.empty(z.shape)
        z.apply_sigmoid(self._sig)
        return self._sig

    def apply_derivative(self, z):
        self._free_memory()

        self._sig = cm.empty(z.shape)
        z.apply_sigmoid(self._sig)

        one_m_sig = cm.empty(self._sig.shape)
        self._sig.mult(-1, one_m_sig)
        one_m_sig.add(1)
        self._sig.mult(one_m_sig)
        del one_m_sig

        return self._sig

    def _free_memory(self):
        if hasattr(self, '_sig'):
            del self._sig
