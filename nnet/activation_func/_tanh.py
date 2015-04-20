from __future__ import absolute_import, division

import cudamat as cm

from ._base import ActivationFuncBase


class Tanh(ActivationFuncBase):

    def apply(self, z):
        self._free_memory()

        self._th = cm.empty(z.shape)
        z.apply_tanh(self._th)
        return self._th

    def apply_derivative(self, z):
        self._free_memory()

        self._th = cm.empty(z.shape)
        z.apply_tanh(self._th)

        cm.pow(self._th, 2, self._th)
        self._th.mult(-1).add(1)
        return self._th

    def _free_memory(self):
        if hasattr(self, '_th'):
            del self._th
