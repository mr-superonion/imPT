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

# You can define your own observable function

MISSING = "if_you_see_this_there_was_a_mistake_creating_an_observable"


class Observable(object):
    def __init__(self, **kwargs):
        self.meta = {}
        self.meta.update(**kwargs)
        self._set_obs_func(self.this_func)
        return

    def _set_obs_func(self, func):
        self._obs_func = func

    def this_func(*args):
        raise RuntimeError("Your observable code needs to "
            "over-ride the this_func method so it knows how to "
            "load the observed data"
            )

    def __call__(self, x):
        if not hasattr(x, "apply"):
            raise TypeError("The input object is not a dask DataFrame")
        return x.apply(self._obs_func, axis=1, meta=(None, '<f8'))

    def __add__(self, other):
        obs = Observable()
        obs.meta = self.meta.update(other.meta)
        func = lambda x: self._obs_func(x) + other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __sub__(self, other):
        obs = Observable()
        obs.meta = self.meta.update(other.meta)
        func = lambda x: self._obs_func(x) - other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __mul__(self, other):
        obs = Observable()
        obs.meta = self.meta.update(other.meta)
        func = lambda x: self._obs_func(x) * other._obs_func(x)
        obs._set_obs_func(func)
        return obs

    def __truediv__(self, other):
        obs = Observable()
        obs.meta = self.meta.update(other.meta)
        func = lambda x: self._obs_func(x) / other._obs_func(x)
        obs._set_obs_func(func)
        return obs



# Users can follow the following examples to define their own observales

class fpfs_e1_Li2018(Observable):
    def __init__(self, Const):
        super(fpfs_e1_Li2018, self).__init__(Const=Const)
        self._set_obs_func(self.this_func)
        return

    def this_func(self, x):
        e1 = x["fpfs_M22c"] \
            / ( x["fpfs_M00"] + self.meta["Const"])
        return e1


class fpfs_e2_Li2018(Observable):
    def __init__(self, Const):
        super(fpfs_e2_Li2018, self).__init__(Const=Const)
        self._set_obs_func(self.this_func)
        return

    def this_func(self, x):
        e2 = x["fpfs_M22s"] \
            / ( x["fpfs_M00"] + self.meta["Const"])
        return e2


class fpfs_w_Li2022(Observable):

    def __init__(self, **kwargs):
        self.meta.update(**kwargs)
        return

    def _obs_func(self, x):
        print(MISSING)
        pass
