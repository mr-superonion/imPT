# gimpt autodiff pipline
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

# This file contains pytrees for linear observables measured from images
# and functions to get their shear response

from jax import jit
import jax.numpy as jnp
from functools import partial
from fitsio import read as fitsread
import numpy.lib.recfunctions as rfn

from .default import *
from ..base import LinRespBase


__all__ = ["read_catalog", "FpfsLinResponse"]


"""
The following Classes are for FPFS. Feel free to extend the following system
or take it as an example to develop new system
"""
## TODO: Contact me if you are interested in adding or developing a new system
## of Observables
def read_catalog(fname):
    x = fitsread(fname)[col_names]
    out = rfn.structured_to_unstructured(x, copy=False)
    out = jnp.array(out, dtype=jnp.float64)
    return out

class FpfsLinResponse(LinRespBase):
    @partial(jit, static_argnums=(0,))
    def _dg1(self, row):
        """Returns shear response array [first component] of shapelet pytree"""
        # shear response for shapelet modes
        M00 = -jnp.sqrt(2.0) * row[m22c]
        M20 = -jnp.sqrt(6.0) * row[m42c]
        M22c = (row[m00] - row[m40]) / jnp.sqrt(2.0)
        # TODO: Include spin-4 term. Will add it when we have M44
        M22s = jnp.zeros_like(row[m22s])
        # TODO: Incldue the shear response of M40 in the future. This is not
        # required in the FPFS shear estimation (v1~v3), so I set it to zero
        # here (But if you are interested in playing with shear response of
        # this term, please contact me.)
        M40 = jnp.zeros_like(row[m40])
        M42c = jnp.zeros_like(row[m42c])
        M42s = jnp.zeros_like(row[m42s])
        out = jnp.stack(
            [
                M00,
                M20,
                M22c,
                M22s,
                M40,
                M42c,
                M42s,
                row[v0_g1],
                row[v1_g1],
                row[v2_g1],
                row[v3_g1],
                row[v4_g1],
                row[v5_g1],
                row[v6_g1],
                row[v7_g1],
            ] + [0] * 16
        )
        return out

    @partial(jit, static_argnums=(0,))
    def _dg2(self, row):
        """Returns shear response array [second component] of shapelet pytree"""
        M00 = -jnp.sqrt(2.0) * row[m22s]
        M20 = -jnp.sqrt(6.0) * row[m42s]
        # TODO: Include spin-4 term. Will add it when we have M44
        M22c = jnp.zeros_like(row[m22c])
        M22s = (row[m00] - row[m40]) / jnp.sqrt(2.0)
        # TODO: Incldue the shear response of M40 in the future. This is not
        # required in the FPFS shear estimation (v1~v3), so I set it to zero
        # here (But if you are interested in playing with shear response of
        # this term, please contact me.)
        M40 = jnp.zeros_like(row[m40])
        M42c = jnp.zeros_like(row[m42c])
        M42s = jnp.zeros_like(row[m42s])
        out = jnp.stack(
            [
                M00,
                M20,
                M22c,
                M22s,
                M40,
                M42c,
                M42s,
                row[v0_g2],
                row[v1_g2],
                row[v2_g2],
                row[v3_g2],
                row[v4_g2],
                row[v5_g2],
                row[v6_g2],
                row[v7_g2],
            ] + [0] * 16
        )
        return out
