# impt autodiff pipline
# Copyright 20221114 Xiangchong Li.
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
import numpy as np
from jax import jit
import jax.numpy as jnp


from .default import col_names, ncol

__all__ = ["fpfscov_to_imptcov"]


def fpfscov_to_imptcov(data):
    """Converts FPFS noise Covariance elements into a covariance matrix of
    lensPT.

    Args:
        data (ndarray):     input FPFS ellipticity catalog
    Returns:
        out (ndarray):      Covariance matrix
    """
    # the colum names
    # M00 -> N00; v1 -> V1
    ll = [cn[5:].replace("M", "N").replace("v", "V") for cn in col_names]
    out = np.zeros((ncol, ncol))
    for i in range(ncol):
        for j in range(ncol):
            try:
                try:
                    cname = "fpfs_%s%s" % (ll[i], ll[j])
                    out[i, j] = data[cname][0]
                except ValueError:
                    cname = "fpfs_%s%s" % (ll[j], ll[i])
                    out[i, j] = data[cname][0]
            except ValueError:
                out[i, j] = 0.0
    out = jnp.array(out)
    return out


@jit
def tsfunc(x, mu, sigma):
    """Returns the weight funciton.
    This is for C2 sinusoidal based funciton

    Args:
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (float):  width of the selection function
    Returns:
        out (ndarray):  the weight funciton
    """
    t = (x - mu) / sigma

    def func(t):
        return 1.0 / 2.0 + t / 2.0 + 1.0 / 2.0 / jnp.pi * jnp.sin(t * jnp.pi)

    return jnp.piecewise(t, [t < -1, (t >= -1) & (t <= 1), t > 1], [0.0, func, 1.0])


@jit
def sigfunc(x, mu=0.0, sigma=1.5):
    """Returns the weight funciton [deriv=0], or the *multiplicative factor* to
    the weight function for first order derivative [deriv=1]

    Args:
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (float):  width of the selection function
    Returns:
        out (ndarray):  the weight funciton [deriv=0], or the *multiplicative
                        factor* to the weight function for first order derivative
                        [deriv=1]
    """
    expx = np.exp(-(x - mu) / sigma)
    # sigmoid function
    return 1.0 / (1.0 + expx)
