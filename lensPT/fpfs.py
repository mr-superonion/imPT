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

# This is only a simple example of shear estimator
# You can define your own observable function


import jax.numpy as jnp
from .observable import Observable

def tsfunc2(x, mu=0.0, sigma=1.5):
    """Returns the weight funciton [deriv=0], or the *multiplicative factor* to
    the weight function for first order derivative [deriv=1]. This is for C2
    funciton

    Args:
        deriv (int):    whether do derivative [deriv=1] or not [deriv=0]
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (float):  width of the selection function
    Returns:
        out (ndarray):  the weight funciton [deriv=0], or the *multiplicative
                        factor* to the weight function for first order
                        derivative [deriv=1]
    """
    t = (x - mu) / sigma
    def func(t):
        return 1.0 / 2.0 + t / 2.0 + 1.0 / 2.0 / jnp.pi * jnp.sin(t * jnp.pi)
    return jnp.piecewise(t, [t < -1, (t >= -1) & (t <= 1), t > 1], [0.0, func, 1.0])

def fpfs_basis_mapping(basis_name,x):
    if basis_name == "fpfs_M22c":
        return jnp.zeros_like(x)
    elif basis_name == "fpfs_M22s":
        return jnp.zeros_like(x)
    elif basis_name == "fpfs_M20":
        return jnp.zeros_like(x)
    elif basis_name == "fpfs_M00":
        return jnp.zeros_like(x)
    elif basis_name == "fpfs_M00":
        return jnp.zeros_like(x)
    elif basis_name == "fpfs_M20":
        return jnp.zeros_like(x)
    elif basis_name == "fpfs_M40":
        # TODO: XL: Incldue the shear response of M40 in the future. This is
        # not required in the FPFS shear estimation (v1~v3), so I set it to
        # zero here (If you are interested in it, please contact me.)
        return jnp.zeros_like(x)
    else:
        raise ValueError("basis_name: %s is not supported" %basis_name)


class E1(Observable):
    def __init__(self, Const):
        super(E1, self).__init__(Const=Const)
        self.mode_names = [
            "fpfs_M22c",
            "fpfs_M00",
            "fpfs_M40",
        ]
        self.nmodes = len(self.mode_names)
        self.has_dg = True
        return

    def _base_func(self, x):
        e1 = x[self.aind("fpfs_M22c")] / \
                (x[self.aind("fpfs_M00")] + self.meta["Const"])
        return e1

    def _dm_dg1(self, x):
        dM22c = 1. / jnp.sqrt(2.) * \
                (x[self.aind("fpfs_M00")] - x[self.aind("fpfs_M40")])
        dM00 = -jnp.sqrt(2.) * x[self.aind("fpfs_M22c")]
        return jnp.array([dM22c, dM00, 0.])

    def _dm_dg2(self, x):
        """This is spin-4 part, which is set to zero (rotational symmetry)
        """
        return jnp.zeros(self.nmodes)


class E2(Observable):
    def __init__(self, Const):
        super(E2, self).__init__(Const=Const)
        self.mode_names = [
            "fpfs_M22s",
            "fpfs_M00",
            "fpfs_M40",
        ]
        self.nmodes = len(self.mode_names)
        self.has_dg = True
        return

    def _base_func(self, x):
        e1 = x[self.aind("fpfs_M22s")] / (x[self.aind("fpfs_M00")] + self.meta["Const"])
        return e1

    def _dm_dg1(self, x):
        """This is spin-4 part, which is set to zero (rotational symmetry)
        """
        return jnp.zeros(self.nmodes)

    def _dm_dg2(self, x):
        dM22s = 1. / jnp.sqrt(2.) * \
                (x[self.aind("fpfs_M00")] - x[self.aind("fpfs_M40")])
        dM00 = -jnp.sqrt(2.) * x[self.aind("fpfs_M22s")]
        return jnp.array([dM22s, dM00, 0.])


class Weight(Observable):
    def __init__(self, **kwargs):
        super(Weight, self).__init__()
        self.mode_names = [
            "fpfs_M00",
            "fpfs_M40",
            "fpfs_M22s",
        ]
        for _ in range(8):
            self.mode_names.append("fpfs_v%d" %_)
            self.mode_names.append("fpfs_v%dr1" %_)
            self.mode_names.append("fpfs_v%dr2" %_)
        self.nmodes = len(self.mode_names)
        self.has_dg = True
        return

    def _base_func(self, x):
        pass
