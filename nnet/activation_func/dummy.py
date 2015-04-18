from __future__ import absolute_import, division

import numpy as np

from .base import ActivationFuncBase
from ..neural_net_exception import NeuralNetException

class Dummy(ActivationFuncBase):
    """
    For input layer.
    """

    def apply(self, z):
        return z

    def apply_derivative(self, z):
        raise NeuralNetException('apply_derivative is called on dummy layer.')