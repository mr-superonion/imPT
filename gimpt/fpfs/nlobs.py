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

# This file contains modules for nonlinear observables measured from images
from flax import struct
import jax.numpy as jnp

from .default_fpfs import *
from ..base import NlBase

__all__ = ["FpfsE1", "FpfsE2", "FpfsParams", "FpfsNodeParams"]

"""
The following Classes are for FPFS. Feel free to extend the following system
or take it as an example to develop new system
"""
## TODO: Contact me if you are interested in developing a new system of
## Observables

class FpfsParams(struct.PyTreeNode):
    """FPFS parameter tree, fixed parameter"""
    Const: jnp.float64 = struct.field(pytree_node=False, default=10.0)

class FpfsNodeParams(struct.PyTreeNode):
    """FPFS parameter tree, unfixed parameter, used for training"""
    Const: jnp.float64 = struct.field(pytree_node=True, default=10.0)

class FpfsObsBase(NlBase):
    def __init__(self, params, parent=None):
        if not isinstance(params, FpfsParams):
            raise TypeError("params is not FPFS parameters")
        super().__init__(params, parent)

class FpfsE1(FpfsObsBase):
    def _base_func(self, cat):
        return cat[m22c] / (cat[m00] + self.params.Const)

class FpfsE2(FpfsObsBase):
    def _base_func(self, cat):
        return cat[m22s] / (cat[m00] + self.params.Const)
