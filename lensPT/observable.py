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


import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev, grad
import numpy.lib.recfunctions as rfn

MISSING = "if_you_see_this_there_was_a_mistake_creating_an_observable"


class Observable(object):
    def __init__(self, **kwargs):
        self.meta = {}
        self.meta.update(**kwargs)
        self.mode_names = [MISSING]
        self._set_obs_func(self.base_func)
        self._obs_hessian_func = jacfwd(jacrev(self._obs_func))
        self._obs_grad_func = grad(self._obs_func)
        return

    def _set_obs_func(self, func):
        self._obs_func = func

    def base_func(*args):
        raise RuntimeError("Your observable code needs to "
            "over-ride the base_func method so it knows how to "
            "load the observed data"
            )

    def prepare_array(self,x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input array should be structured array")
        if x.dtype.names is None:
            return x
        else:
            return rfn.structured_to_unstructured(x[self.mode_names], copy=False)

    def aind(self, colname):
        return self.mode_names.index(colname)

    def _make_new(self,other):
        obs = Observable()
        obs.meta = self.meta.update(other.meta)
        obs.mode_names = list(set(self.mode_names) | set(other.mode_names))
        return obs

    def __add__(self, other):
        obs = self._make_new(other)
        func = lambda x: self._obs_func(x) + other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __sub__(self, other):
        obs = self._make_new(other)
        func = lambda x: self._obs_func(x) - other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __mul__(self, other):
        obs = self._make_new(other)
        func = lambda x: self._obs_func(x) * other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __truediv__(self, other):
        obs = self._make_new(other)
        func = lambda x: self._obs_func(x) / other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def evaluate(self, x):
        """Calls this observable function"""
        x = self.prepare_array(x)
        return jnp.apply_along_axis(self._obs_func, axis=-1, arr=x)

    def grad(self, x):
        """Calls the gradient vector function of observable function"""
        x = self.prepare_array(x)
        return jnp.apply_along_axis(self._obs_grad_func, axis=-1, arr=x)

    def hessian(self, x):
        """Calls the hessian matrix function of observable function"""
        x = self.prepare_array(x)
        return jnp.apply_along_axis(self._obs_hessian_func, axis=-1, arr=x)



# Users can follow the following examples to define their own observales

class fpfs_e1_Li2018(Observable):
    def __init__(self, Const):
        super(fpfs_e1_Li2018, self).__init__(Const=Const)
        self._set_obs_func(self.base_func)
        self.mode_names = [
                "fpfs_M22c",
                "fpfs_M00",
                ]
        return

    def base_func(self, x):
        e1 = x[self.aind("fpfs_M22c")] \
            / ( x[self.aind("fpfs_M00")] + self.meta["Const"] )
        return e1


class fpfs_e2_Li2018(Observable):
    def __init__(self, Const):
        super(fpfs_e2_Li2018, self).__init__(Const=Const)
        self._set_obs_func(self.base_func)
        self.mode_names = [
                "fpfs_M22s",
                "fpfs_M00",
                ]
        return

    def base_func(self, x):
        e1 = x[self.aind("fpfs_M22s")] \
            / ( x[self.aind("fpfs_M00")] + self.meta["Const"])
        return e1


class fpfs_w_Li2022(Observable):

    def __init__(self, **kwargs):
        self.meta.update(**kwargs)
        return

    def _obs_func(self, x):
        pass
