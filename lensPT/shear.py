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


class Gperturb1(Observable):
    """A Base Class to derive the first-order shear perturbation
    for an observable."""

    def __init__(self, obs_obj):
        """Initializes shear response object using a obs_obj object and
        a noise covariance matrix
        """
        super(Gperturb1, self).__init__()
        if not hasattr(obs_obj, "grad"):
            raise ValueError("obs_fun does not has gradient")
        if obs_obj.dg_obj is None:
            raise ValueError("input obs_obj should have shear response")
        self.update_obs(obs_obj)
        return

    def update_obs(self, obs_obj):
        """Updates the observable funciton."""
        self.obs_obj = obs_obj
        self.mode_names = obs_obj.mode_names
        return

    def _make_new(self):
        obs = Observable()
        obs.mode_names = list(set(self.obs_obj.mode_names)
                | set(self.obs_obj.mode_names))
        return obs


class g1_perturb1(Gperturb1):
    """A Functional Class to derive the first-order shear perturbation
    for an observable function."""

    def __init__(self, obs_obj):
        """Initializes shear response object using an ObsObject
        """
        super(g1_perturb1, self).__init__(obs_obj)
        return

    def _base_func(self, x):
        """Returns the first-order shear response"""
        res = jnp.dot(self.obs_obj._obs_grad_func(x), self._dm_dg(x))
        return res

    def _dm_dg(self, x):
        out = self.obs_obj.dg_obj.dm_dg(
            x, self.mode_names, 1
        )
        return out


class g2_perturb1(Gperturb1):
    """A Functional Class to derive the first-order shear perturbation for an
    observable function."""

    def __init__(self, obs_obj):
        """Initializes shear response object using an ObsObject
        """
        super(g2_perturb1, self).__init__(obs_obj)
        return

    def _base_func(self, x):
        """Returns the first-order shear response"""
        res = jnp.dot(self.obs_obj._obs_grad_func(x), self._dm_dg(x))
        return res

    def _dm_dg(self, x):
        out = self.obs_obj.dg_obj.dm_dg(
            x, self.mode_names, 2
        )
        return out
