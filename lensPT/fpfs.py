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


class shapelets_shear(object):
    def __init__(self, **kwargs):
        super(shapelets_shear, self).__init__()
        self.initialize_meta(**kwargs)
        return

    def initialize_meta(self, **kwargs):
        try:
            self.meta
        except AttributeError:
            self.meta = {}
        self.meta.update(**kwargs)
        try:
            self.meta2
        except AttributeError:
            self.meta2 = {
                "modes_tmp": [],  # used to call a funciton
            }
        self.meta2.update(**kwargs)
        return

    def aind(self, colname):
        return self.meta2["modes_tmp"].index(colname)

    def _dm_dg1(self, x, basis_name):
        if basis_name == "fpfs_M00":
            out = -jnp.sqrt(2.0) * x[self.aind("fpfs_M22c")]
        elif basis_name == "fpfs_M20":
            out = -jnp.sqrt(6.0) * x[self.aind("fpfs_M42c")]
        elif basis_name == "fpfs_M22c":
            out = (x[self.aind("fpfs_M00")] - x[self.aind("fpfs_M40")]) / jnp.sqrt(2.0)
        elif basis_name == "fpfs_M22s":
            # TODO: Neglect spin-4 term. Need to add it when we have M44
            out = 0.0
        elif basis_name == "fpfs_M40":
            # NOTE: Incldue the shear response of M40 in the future. This is not
            # required in the FPFS shear estimation (v1~v3), so I set it to zero
            # here (But if you are interested in playing with shear response of
            # this term, please contact me.)
            out = 0.0
        else:
            out = 0.0
        return out

    def _dm_dg2(self, x, basis_name):
        if basis_name == "fpfs_M00":
            out = -jnp.sqrt(2.0) * x[self.aind("fpfs_M22s")]
        elif basis_name == "fpfs_M20":
            out = -jnp.sqrt(6.0) * x[self.aind("fpfs_M42s")]
        elif basis_name == "fpfs_M22c":
            # TODO: Neglect spin-4 term. Need to add it when we have M44
            out = 0.0
        elif basis_name == "fpfs_M22s":
            out = (x[self.aind("fpfs_M00")] - x[self.aind("fpfs_M40")]) / jnp.sqrt(2.0)
        elif basis_name == "fpfs_M40":
            # NOTE: Incldue the shear response of M40 in the future. This is not
            # required in the FPFS shear estimation (v1~v3), so I set it to zero
            # here (But if you are interested in playing with shear response of
            # this term, please contact me.)
            out = 0.0
        else:
            out = 0.0
        return out

    def dm_dg(self, data, name_list, g_comp):
        """Returns shear response of shapelet basis

        Args:
            data (ndarray):     multi-row array
            name_list (list):   a list of name of the shapelet basis
            g_comp (int):       the component of shear [1 or 2]
        Returns:
            out (ndarray):      shear responses for the shapelet bases
        """
        if g_comp == 1:

            def _func_(x, basis_name):
                return jnp.apply_along_axis(
                    func1d=self._dm_dg1,
                    axis=-1,
                    arr=x,
                    basis_name=basis_name,
                )

        elif g_comp == 2:

            def _func_(x, basis_name):
                return jnp.apply_along_axis(
                    func1d=self._dm_dg2,
                    axis=-1,
                    arr=x,
                    basis_name=basis_name,
                )

        else:
            raise ValueError("g_comp can only be 1 or 2")
        out = jnp.array([_func_(x=data, basis_name=nm) for nm in name_list]).T
        return out


class weighted_e1(Observable, shapelets_shear):
    def __init__(self, Const):
        super(weighted_e1, self).__init__(Const=Const)
        self.umode_names = None
        self.meta["modes"] = [
            "fpfs_M22c",
            "fpfs_M00",
        ]
        self.nmodes = len(self.meta["modes"])
        # NOTE: XL: I manually put dmode_names, which I know is not clever;
        # Will make a dictionary for that
        self.meta["modes_child"] = [
            "fpfs_M22c",
            "fpfs_M00",
            "fpfs_M40",
        ]
        return

    def _base_func(self, x):
        out = x[self.aind("fpfs_M22c")] / (
            x[self.aind("fpfs_M00")] + self.meta["Const"]
        )
        return out


class weighted_e2(Observable, shapelets_shear):
    def __init__(self, Const):
        super(weighted_e2, self).__init__(Const=Const)
        self.umode_names = None
        self.meta["modes"] = [
            "fpfs_M22s",
            "fpfs_M00",
        ]
        self.nmodes = len(self.meta["modes"])
        # NOTE: XL: I manually put dmode_names, which I know is not clever;
        # Will make a dictionary for that
        self.meta["modes_child"] = [
            "fpfs_M22s",
            "fpfs_M00",
            "fpfs_M40",
        ]
        return

    def _base_func(self, x):
        out = x[self.aind("fpfs_M22s")] / (
            x[self.aind("fpfs_M00")] + self.meta["Const"]
        )
        return out


# class peak_weight(Observable):
#     def __init__(self, **kwargs):
#         super(peak_weight, self).__init__()
#         self.mode_names = [
#             "fpfs_M00",
#             "fpfs_M40",
#             "fpfs_M22s",
#         ]
#         for _ in range(8):
#             self.mode_names.append("fpfs_v%d" %_)
#             self.mode_names.append("fpfs_v%dr1" %_)
#             self.mode_names.append("fpfs_v%dr2" %_)
#         self.nmodes = len(self.mode_names)
#         self.has_dg = True
#         return

#     def _base_func(self, x):
#         pass
