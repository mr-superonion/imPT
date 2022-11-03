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


class noise_bias_perturb2nd(object):
    """A Functional Class to derive the second-order noise perturbation
    function."""

    def __init__(self, obs_func, noise_cov):
        """Initializes noise bias function object using a obs_func object and
        a noise covariance matrix
        """
        if not hasattr(obs_func, "evaluate"):
            raise ValueError("obs_fun does not has evaluation")
        if not hasattr(obs_func, "hessian"):
            raise ValueError("obs_fun does not has hessian")
        self.update_all(obs_func, noise_cov)
        return

    def update_noise_cov(self, noise_cov):
        """Updates the noise covariance"""
        self.noise_cov = noise_cov
        return

    def update_all(self, obs_func, noise_cov):
        """Updates the observable funciton and the noise covariance"""
        self._obs_func_obj = obs_func
        self.update_noise_cov(noise_cov)
        return

    def check_vector(self, x):
        """checks whether a data vector meets the requirements"""
        ndata = x.shape[-1]
        if self.noise_cov.shape != (ndata, ndata):
            raise ValueError(
                "input data should have length %d" % self.noise_cov.shape[0]
            )

    def _noise_bias_func(self, x):
        indexes = [[-2, -1], [-2, -1]]
        b = jnp.tensordot(self._obs_func_obj._obs_hessian_func(x),
                self.noise_cov, indexes) / (-2.0)
        return b

    def evaluate(self, x):
        """Evaluate the noise bias funciton"""
        x = self._obs_func_obj.prepare_array(x)
        return jnp.apply_along_axis(self._noise_bias_func, axis=-1, arr=x)
