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

# You can define your own observable function


import fitsio
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev, grad
import numpy.lib.recfunctions as rfn


MISSING = "if_you_see_this_there_was_a_mistake_creating_an_observable"

__all__ = ["Observable", "Catalog", "prepare_array"]


def prepare_array(x, colnames=None):
    """Prepare a unstructured array from structrued array [read by fitsio]

    Args:
        x (ndarray):        structured ndarray
        colnames (list):    a list of column names
    Returns:
        out (ndarray):      unstructured ndarray
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input array should be structured array")
    if x.dtype.names is None:
        out = x
    else:
        out = rfn.structured_to_unstructured(x[colnames], copy=False)
        out = jnp.array(out, dtype=jnp.float64)
    return out


class Observable(object):
    def __init__(self, **kwargs):
        super(Observable, self).__init__()
        self.initialize_meta(**kwargs)
        self._set_obs_func(self._base_func)
        self.parent_obj = None
        return

    def initialize_meta(self, **kwargs):
        try:
            self.meta
        except AttributeError:
            self.meta = {
                "modes": [],  # current funciton modes
                "modes_child": [],  # next-order funciton modes [dg]
                "modes_parent": [],  # previous-order funciton modes [int g]
            }
        self.meta.update(**kwargs)
        try:
            self.meta2
        except AttributeError:
            self.meta2 = {
                "modes_tmp": [],  # used to call a funciton
            }
        self.distort = None
        return

    def _set_obs_func(self, func):
        """Setup observable functions [func, derivative and Hessian]
        """
        self._obs_func = func
        self._obs_grad_func = grad(self._obs_func)
        self._obs_hessian_func = jacfwd(jacrev(self._obs_func))
        return

    def _base_func(*args):
        raise RuntimeError(
            "Your observable code needs to "
            "over-ride the _base_func method so it knows how to "
            "load the observed data"
        )

    def _make_obs_new(self, other):
        obs = Observable()
        obs.meta = self.meta.copy()
        for kk in ["modes_parent", "modes_child", "modes"]:
            obs.meta[kk] = list(set(obs.meta[kk]) | set(other.meta[kk]))
        for kk in other.meta.keys():
            if kk not in obs.meta.keys():
                obs.meta[kk] = other.meta[kk]
        self.meta2 = other.meta2
        obs.meta2 = self.meta2
        obs.distort = self.distort
        obs.distort.meta2 = obs.meta2
        return obs

    def make_metas_child(self):
        meta = self.meta.copy()
        meta["modes"] = self.meta["modes_child"]
        meta["modes_parent"] = self.meta["modes"]
        meta["modes_child"] = []
        meta2 = self.meta2
        return meta, meta2

    def __add__(self, other):
        obs = self._make_obs_new(other)
        func = lambda x: self._obs_func(x) + other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __sub__(self, other):
        obs = self._make_obs_new(other)
        func = lambda x: self._obs_func(x) - other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __mul__(self, other):
        obs = self._make_obs_new(other)
        func = lambda x: self._obs_func(x) * other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __truediv__(self, other):
        obs = self._make_obs_new(other)
        func = lambda x: self._obs_func(x) / other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def aind(self, colname):
        return self.meta2["modes_tmp"].index(colname)

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
        return

    def _postcompute(self):
        # back to empty modes_tmp
        self.meta2["modes_tmp"] = []
        return

    def evaluate(self, cat):
        """Calls this observable function"""
        self._precompute(cat)
        out = jnp.apply_along_axis(func1d=self._obs_func, axis=-1, arr=cat.data)
        self._postcompute()
        return out

    def grad(self, cat):
        """Calls the gradient vector function of observable function"""
        self._precompute(cat)
        out = jnp.apply_along_axis(func1d=self._obs_grad_func, axis=-1, arr=cat.data)
        self._postcompute()
        return out

    def hessian(self, cat):
        """Calls the hessian matrix function of observable function"""
        self._precompute(cat)
        out = jnp.apply_along_axis(func1d=self._obs_hessian_func, axis=-1, arr=cat.data)
        self._postcompute()
        return out


class Catalog(object):
    """A class for catalogs of measurements from images"""

    def __init__(self, mode_names, data, noise_cov=None):
        if isinstance(data, str):
            data = fitsio.read(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data should be str or ndarray")
        self.mode_names = mode_names
        self.data = prepare_array(data, self.mode_names)

        if noise_cov is not None:
            if isinstance(noise_cov, str):
                noise_cov = jnp.array(fitsio.read(noise_cov))
            if not isinstance(noise_cov, jnp.ndarray):
                raise TypeError("Input noise covariance should be str or np.ndarray")
            ncol = len(mode_names)
            if not noise_cov.shape == (ncol, ncol):
                raise ValueError("input noise cov has an inconsistent shape")
            self.noise_cov = noise_cov
        else:
            self.noise_cov = None
        return
