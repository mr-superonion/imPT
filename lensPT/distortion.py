# lensPT autodiff pipline
# Copyright 20221031 Xiangchong Li.
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

import jax.numpy as jnp
from .observable import Observable

__all__ = ["g1Perturb1", "g2Perturb1"]


class Gperturb1(Observable):
    """A Base Class to derive the first-order shear perturbation
    for an observable. The perturbation follows eq. (4) of
    https://arxiv.org/abs/2208.10522
    """

    def __init__(self, parent_obj):
        """Initializes shear response object using a parent_obj object and
        a noise covariance matrix
        """
        super(Gperturb1, self).__init__()
        if not hasattr(parent_obj, "grad"):
            raise ValueError("obs_fun does not has gradient")
        if not hasattr(parent_obj, "dm_dg"):
            raise ValueError("input parent_obj should have shear response")
        self.update_parent(parent_obj)
        return

    def update_parent(self, parent_obj):
        """Updates the observable funciton with a parent parent_obj."""
        meta, meta2 = parent_obj.make_metas_child()
        self.meta = meta
        self.meta2 = meta2
        self.parent_obj = parent_obj
        return

    def _base_func(self, x):
        """Returns the first-order shear response"""
        res = jnp.dot(self.parent_obj._obs_grad_func(x), self._dm_dg(x))
        return res

    def _dm_dg(*args):
        raise RuntimeError(
            "Your observable code needs to over-ride the _dm_dg method "
            "in the shear pserturbation class so it knows how to compute "
            "shear responses of basis modes"
        )


class g1Perturb1(Gperturb1):
    """A Functional Class to derive the first-order shear [the first component]
    perturbation for an observable function.
    """

    def __init__(self, parent_obj):
        """Initializes shear response object using an ObsObject"""
        super(g1Perturb1, self).__init__(parent_obj)
        return

    def _dm_dg(self, x):
        out = self.parent_obj.dm_dg(x, self.meta2["modes_tmp"], 1)
        return out


class g2Perturb1(Gperturb1):
    """A Functional Class to derive the first-order shear [the second
    component] perturbation for an observable function.
    """

    def __init__(self, parent_obj):
        """Initializes shear response object using an ObsObject"""
        super(g2Perturb1, self).__init__(parent_obj)
        return

    def _dm_dg(self, x):
        out = self.parent_obj.dm_dg(x, self.meta2["modes_tmp"], 2)
        return out
