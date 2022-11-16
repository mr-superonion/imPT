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

__all__ = ["G1Perturb1", "G2Perturb1"]


class Gperturb1(Observable):
    """A Base Class to derive the first-order shear perturbation
    for an observable. The perturbation follows eq. (4) of
    https://arxiv.org/abs/2208.10522
    """

    def __init__(self, parent_obj):
        """Initializes shear response object using a parent_obj object and
        a noise covariance matrix.
        """
        super(Gperturb1, self).__init__()
        if not hasattr(parent_obj, "grad"):
            raise ValueError("obs_fun does not has gradient")
        if not hasattr(parent_obj, "distort"):
            raise ValueError("input parent_obj should have shear response")
        if not hasattr(parent_obj.distort, "dm_dg"):
            raise ValueError("input parent_obj should have shear response")
        self.initialize_with_parent(parent_obj)
        self.ig = 0
        return

    def initialize_with_parent(self, parent_obj):
        """Updates the observable funciton with a parent parent_obj."""
        meta, meta2 = parent_obj.make_metas_child()
        self.meta = meta
        self.meta2 = meta2
        self.parent_obj = parent_obj
        self.distort = parent_obj.distort.make_child()
        return

    def _base_func(self, x):
        """Returns the first-order shear response."""
        res = jnp.dot(
            self.parent_obj._obs_grad_func(x),
            self.parent_obj.distort.dm_dg(x, self.meta2["modes_tmp"], self.ig),
        )
        return res


class G1Perturb1(Gperturb1):
    """A Functional Class to derive the first-order shear [the first component]
    perturbation for an observable function.
    """

    def __init__(self, parent_obj):
        """Initializes shear response object using an ObsObject."""
        super(G1Perturb1, self).__init__(parent_obj)
        self.ig = 1
        return


class G2Perturb1(Gperturb1):
    """A Functional Class to derive the first-order shear [the second
    component] perturbation for an observable function.
    """

    def __init__(self, parent_obj):
        """Initializes shear response object using an ObsObject."""
        super(G2Perturb1, self).__init__(parent_obj)
        self.ig = 2
        return
