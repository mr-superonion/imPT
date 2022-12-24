# gimpt autodiff pipline
# Copyright 20221222 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# python lib

# This file contains modules for perturbation functionals
# (vector perturbation: shear & tensor perturbation: noise)

import jax.numpy as jnp
from jax import jit

from .base import NlBase

__all__ = ["RespG1", "RespG2"]


class RespG1(NlBase):
    """A Base Class to derive the first-order  shear response [first component]
    for an observable, following eq. (4) of
    https://arxiv.org/abs/2208.10522
    """

    def __init__(self, parent):
        """Initializes shear response object using a parent_obj object and
        a noise covariance matrix.
        """
        if not hasattr(parent, "_obs_grad_func"):
            raise TypeError("parent object does not has gradient operation")
        super().__init__(parent.params, parent, parent.linResp)
        return

    def _base_func(self, x):
        """Returns the first-order shear response."""
        res = jnp.dot(
            self.parent._obs_grad_func(x),
            self.linResp._dg1(x),
        )
        return res

class RespG2(NlBase):
    """A Base Class to derive the first-order  shear response [second component]
    for an observable, following eq. (4) of
    https://arxiv.org/abs/2208.10522
    """

    def __init__(self, parent):
        """Initializes shear response object using a parent_obj object and
        a noise covariance matrix.
        """
        if not hasattr(parent, "_obs_grad_func"):
            raise TypeError("parent object does not has gradient operation")
        super().__init__(parent.params, parent, parent.linResp)
        return

    @jit
    def _base_func(self, x):
        """Returns the first-order shear response."""
        res = jnp.dot(
            self.parent._obs_grad_func(x),
            self.linResp._dg2(x),
        )
        return res
