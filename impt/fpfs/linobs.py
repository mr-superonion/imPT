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

# This file contains pytrees for linear observables measured from images
# and functions to get their shear response

# from jax import jit
# from functools import partial
import jax.numpy as jnp
from fitsio import read as fitsread
import numpy.lib.recfunctions as rfn

from .default import indexes as did
from .default import col_names
from ..base import LinRespBase


__all__ = ["read_catalog", "FpfsLinResponse"]


"""
The following Classes are for FPFS. Feel free to extend the following system
or take it as an example to develop new system
"""
# TODO: Contact me if you are interested in adding or developing a new system
# of Observables


def read_catalog(fname):
    x = fitsread(fname)
    if x.dtype.names is not None:
        x = x[col_names]
        x = rfn.structured_to_unstructured(x, copy=False)
    return jnp.array(x, dtype=jnp.float64)


class FpfsLinResponse(LinRespBase):

    # @partial(jit, static_argnums=(0,))
    def _dg1(self, row):
        """Returns shear response array [first component] of shapelet pytree"""
        # shear response for shapelet modes
        fpfs_m00 = -jnp.sqrt(2.0) * row[did["m22c"]]
        fpfs_m20 = -jnp.sqrt(6.0) * row[did["m42c"]]
        fpfs_m22c = (row[did["m00"]] - row[did["m40"]]) / jnp.sqrt(2.0)
        # TODO: Include spin-4 term. Will add it when we have M44
        fpfs_m22s = jnp.zeros_like(row[did["m22s"]])
        # TODO: Incldue the shear response of M40 in the future. This is not
        # required in the FPFS shear estimation (v1~v3), so I set it to zero
        # here (But if you are interested in playing with shear response of
        # this term, please contact me.)
        fpfs_m40 = jnp.zeros_like(row[did["m40"]])
        fpfs_m42c = jnp.zeros_like(row[did["m42c"]])
        fpfs_m42s = jnp.zeros_like(row[did["m42s"]])
        out = jnp.stack(
            [
                fpfs_m00,
                fpfs_m20,
                fpfs_m22c,
                fpfs_m22s,
                fpfs_m40,
                fpfs_m42c,
                fpfs_m42s,
                row[did["v0_g1"]],
                row[did["v1_g1"]],
                row[did["v2_g1"]],
                row[did["v3_g1"]],
                row[did["v4_g1"]],
                row[did["v5_g1"]],
                row[did["v6_g1"]],
                row[did["v7_g1"]],
            ]
            + [0] * 16
        )
        return out

    # @partial(jit, static_argnums=(0,))
    def _dg2(self, row):
        """Returns shear response array [second component] of shapelet pytree"""
        fpfs_m00 = -jnp.sqrt(2.0) * row[did["m22s"]]
        fpfs_m20 = -jnp.sqrt(6.0) * row[did["m42s"]]
        # TODO: Include spin-4 term. Will add it when we have M44
        fpfs_m22c = jnp.zeros_like(row[did["m22c"]])
        fpfs_m22s = (row[did["m00"]] - row[did["m40"]]) / jnp.sqrt(2.0)
        # TODO: Incldue the shear response of M40 in the future. This is not
        # required in the FPFS shear estimation (v1~v3), so I set it to zero
        # here (But if you are interested in playing with shear response of
        # this term, please contact me.)
        fpfs_m40 = jnp.zeros_like(row[did["m40"]])
        fpfs_m42c = jnp.zeros_like(row[did["m42c"]])
        fpfs_m42s = jnp.zeros_like(row[did["m42s"]])
        out = jnp.stack(
            [
                fpfs_m00,
                fpfs_m20,
                fpfs_m22c,
                fpfs_m22s,
                fpfs_m40,
                fpfs_m42c,
                fpfs_m42s,
                row[did["v0_g2"]],
                row[did["v1_g2"]],
                row[did["v2_g2"]],
                row[did["v3_g2"]],
                row[did["v4_g2"]],
                row[did["v5_g2"]],
                row[did["v6_g2"]],
                row[did["v7_g2"]],
            ]
            + [0] * 16
        )
        return out
