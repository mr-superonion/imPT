# FPFS shear estimator
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
from jax import jacfwd, jacrev


class noise_bias_perturb2nd(object):
    """A Class to calculate the second-order noise perturbation"""

    def __init__(self, obs_func, noise_cov):
        """Initializes noise bias function object using a obs_func object and
        a noise covariance matrix
        """
        if not callable(obs_func):
            raise ValueError("obs_fun is not callable")
        self.update_all(obs_func, noise_cov)
        return

    def update_noise_cov(self, noise_cov):
        """Updates the noise covariance"""
        self.noise_cov = noise_cov
        return

    def update_all(self, obs_func, noise_cov):
        """Updates the observable funciton and the noise covariance"""
        self._obs_func = obs_func
        self._obs_hessian_func = jacfwd(jacrev(self._obs_func))
        self.update_noise_cov(noise_cov)
        return

    def obs_func(self, x):
        """Calls the input observable function"""
        return jnp.apply_along_axis(self._obs_func, axis=-1, arr=x)

    def obs_hessian_fun(self, x):
        """Calls the hessian matrix function of observable function"""
        return jnp.apply_along_axis(self._obs_hessian_func, axis=-1, arr=x)

    def check_vector(self, x):
        """checks whether a data vector meets the requirements"""
        ndata = x.shape[-1]
        if self.noise_cov.shape != (ndata, ndata):
            raise ValueError(
                "input data should have length %d" % self.noise_cov.shape[0]
            )

    def _noise_bias_func(self, x):
        indexes = [[-2, -1], [-2, -1]]
        b = jnp.tensordot(self._obs_hessian_func(x), self.noise_cov, indexes) / (-2.0)
        return b

    def __call__(self, x):
        """Applies the noise bias funciton to a numpy array"""
        return jnp.apply_along_axis(self._noise_bias_func, axis=-1, arr=x)
