# impt autodiff pipline
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

# from jax import jit
# from functools import partial
import jax.numpy as jnp
from .base import NlBase

__all__ = ["RespG1", "RespG2", "BiasNoise"]


"""
For shear perturbation, we provide functionals to dervie the two shear response
functions.
This is a mapping from function to function.
"""


class RespG1(NlBase):
    """A Class to derive the shear response function [1st component] for an
    observable, following eq. (4) of
    https://arxiv.org/abs/2208.10522
    """

    def __init__(self, parent):
        """Initializes shear response object using a parent_obj object and
        a noise covariance matrix.
        """
        if not hasattr(parent, "_obs_grad_func"):
            raise TypeError("parent object does not has gradient operation")
        super().__init__(parent.params, parent, parent.lin_resp)
        return

    # @partial(jit, static_argnums=(0,))
    def _base_func(self, x):
        """Returns the first-order shear response."""
        res = jnp.dot(
            self.parent._obs_grad_func(x),
            self.lin_resp._dg1(x),
        )
        return res


class RespG2(NlBase):
    """A Class to derive the shear response function [2nd component] for an
    observable, following eq. (4) of
    https://arxiv.org/abs/2208.10522
    """

    def __init__(self, parent):
        """Initializes shear response object using a parent_obj object and
        a noise covariance matrix.
        """
        if not hasattr(parent, "_obs_grad_func"):
            raise TypeError("parent object does not has gradient operation")
        super().__init__(parent.params, parent, parent.lin_resp)
        return

    # @partial(jit, static_argnums=(0,))
    def _base_func(self, x):
        """Returns the first-order shear response."""
        res = jnp.dot(
            self.parent._obs_grad_func(x),
            self.lin_resp._dg2(x),
        )
        return res


"""
For noise perturbation, we provide a functional to dervie the correction function.
This is a mapping from function to function.
"""


class BiasNoise(NlBase):
    """A Class to derive the second-order noise perturbation function."""

    def __init__(self, parent, noise_cov):
        """Initializes noise response object using a parent_obj object and
        a noise covariance matrix.
        """
        if not hasattr(parent, "_obs_hessian_func"):
            raise TypeError("parent object does not has hessian operation")
        self.update_noise_cov(noise_cov)
        super().__init__(parent.params, parent, parent.lin_resp)
        return

    def update_noise_cov(self, noise_cov):
        self.noise_cov = noise_cov

    # @partial(jit, static_argnums=(0,))
    def _base_func(self, x):
        """Returns the second-order noise response"""
        indexes = [[-2, -1], [-2, -1]]
        res = (
            jnp.tensordot(
                self.parent._obs_hessian_func(x),
                self.noise_cov,
                indexes,
            )
            / 2.0
        )
        return res


class BiasNoiseNull:
    """A Class disabling noise bias correction."""

    def __init__(self):
        """Initializes a null noise response object"""
        self._obs_func = self._base_func
        return

    # @partial(jit, static_argnums=(0,))
    def _base_func(self, x):
        """Returns zero"""
        return 0.0
