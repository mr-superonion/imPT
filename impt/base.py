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

import logging
from flax import struct
from jax import lax, grad, jacfwd, jacrev


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


class LinRespBase:
    def _dg1(*args):
        # each function has a basic function to apply to row and a function
        # wraped with lax
        raise NotImplementedError("You need to over-ride the _dg1 method")

    def _dg2(*args):
        # each function has a basic function to apply to row and a function
        # wraped with lax
        raise NotImplementedError("You need to over-ride the _dg2 method")

    def dg1(self, cat):
        return lax.map(self._dg1, cat)

    def dg2(self, cat):
        return lax.map(self._dg2, cat)


class NlBase:
    def __init__(self, params, parent=None, lin_resp=None):
        self.nmodes = 0
        if parent is not None:
            if not isinstance(parent, NlBase):
                raise ValueError("Input parent is not a instance of NlBase")
        self.parent = parent
        if lin_resp is not None:
            if not isinstance(lin_resp, LinRespBase):
                raise ValueError("Input lin_resp is not a instance of LinRespBase")
        self.lin_resp = lin_resp
        if not isinstance(params, struct.PyTreeNode):
            raise ValueError("Input parameter is not a instance of pyTreeNode")
        self.params = params
        self._set_obs_func(self._base_func)

    def _base_func(*args):
        raise NotImplementedError("You need to over-ride the _base_func method")

    def _set_obs_func(self, func):
        """Setup observable functions [func, derivative and Hessian]"""
        self._obs_func = func
        self._obs_grad_func = grad(self._obs_func, argnums=self.nmodes)
        self._obs_hessian_func = jacfwd(
            jacrev(
                self._obs_func,
            )
        )
        return

    def evaluate(self, cat):
        """Calls this observable function"""
        return lax.map(self._obs_func, cat)

    def grad(self, cat):
        """Calls the gradient vector function of observable function"""
        return lax.map(self._obs_grad_func, cat)

    def hessian(self, cat):
        """Calls the hessian matrix function of observable function"""
        return lax.map(self._obs_hessian_func, cat)

    def make_obs_new(self):
        out = NlBase(self.params, self, self.lin_resp)
        return out

    def __add__(self, other):
        obs = self.make_obs_new()
        if isinstance(other, NlBase):
            func = lambda x: self._obs_func(x) + other._obs_func(x)
        elif isinstance(other, (int, float)):
            func = lambda x: self._obs_func(x) + other
        else:
            raise TypeError("Cannot add %s to observable" % type(other))
        obs._set_obs_func(func)
        return obs

    def __sub__(self, other):
        obs = self.make_obs_new()
        if isinstance(other, NlBase):
            func = lambda x: self._obs_func(x) - other._obs_func(x)
        elif isinstance(other, (int, float)):
            func = lambda x: self._obs_func(x) - other
        else:
            raise TypeError("Cannot subtract %s to observable" % type(other))
        obs._set_obs_func(func)
        return obs

    def __mul__(self, other):
        obs = self.make_obs_new()
        if isinstance(other, NlBase):
            func = lambda x: self._obs_func(x) * other._obs_func(x)
        elif isinstance(other, (int, float)):
            func = lambda x: self._obs_func(x) * other
        else:
            raise TypeError("Cannot multiply %s to observable" % type(other))
        obs._set_obs_func(func)
        return obs

    def __truediv__(self, other):
        obs = self.make_obs_new()
        if isinstance(other, NlBase):
            func = lambda x: self._obs_func(x) / other._obs_func(x)
        elif isinstance(other, (int, float)):
            func = lambda x: self._obs_func(x) / other
        else:
            raise TypeError("Cannot multiply %s to observable" % type(other))
        obs._set_obs_func(func)
        return obs

    def __pow__(self, other):
        obs = self.make_obs_new()
        if isinstance(other, (int, float)):
            func = lambda x: self._obs_func(x) ** other
        else:
            raise TypeError("Cannot power %s to observable" % type(other))
        obs._set_obs_func(func)
        return obs
