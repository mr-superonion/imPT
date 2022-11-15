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

__all__ = ["NoisePerturb2"]


class NoisePerturb2(Observable):
    """A Functional Class to derive the second-order noise perturbation
    function."""

    def __init__(self, obs_obj):
        """Initializes noise bias function object using a obs_obj object and
        a noise covariance matrix
        """
        super(NoisePerturb2, self).__init__()
        if not hasattr(obs_obj, "hessian"):
            raise ValueError("obs_fun does not has hessian")
        self.initialize_with_obs(obs_obj)
        self.noise_cov = None
        return

    def initialize_with_obs(self, obs_obj):
        """Updates the observable funciton and the noise covariance"""
        self.meta = obs_obj.meta
        self.meta2 = obs_obj.meta2
        self.obs_obj = obs_obj
        return

    def _precompute(self, cat):
        # Test whether the input catalog contains all the necessary
        # information
        if not set(self.meta["modes"]).issubset(set(cat.mode_names)):
            raise ValueError(
                "Input catalog does not have all the required\
                    modes"
            )
        # update the modes_tmp
        self.meta2["modes_tmp"] = cat.mode_names
        if cat.noise_cov is None:
            raise AttributeError("Input catalog does not have noise_cov")
        self.noise_cov = cat.noise_cov
        return

    def _postcompute(self):
        # back to empty modes_tmp
        self.meta2["modes_tmp"] = []
        self.noise_cov = None
        return

    def _base_func(self, x):
        """Returns the second-order noise response"""
        indexes = [[-2, -1], [-2, -1]]
        res = (
            jnp.tensordot(
                self.obs_obj._obs_hessian_func(x),
                self.noise_cov,
                indexes,
            )
            / 2.0
        )
        return res
