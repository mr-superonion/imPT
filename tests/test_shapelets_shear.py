# lenspt autodiff pipeline
# Copyright 20221113 Xiangchong Li.
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
"""This unit test checks whether lenspt's shapelets' shear response is
implemented correctly.
"""

import fitsio
import numpy as np
import lenspt as lpt
from jax import config
config.update("jax_enable_x64", True)


ndata = 30
data = fitsio.read("data/fpfs-cut32-0000-g1-0000.fits")
colnames = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
]
ncol = len(colnames)
data2 = lpt.observable.prepare_array(data, colnames)
assert data2.shape == (ndata, ncol), "prepared catalog has incorrect shape"


def test_g1():
    print("testing for shapelets' g1 reponses")
    shapelet_shear = lpt.fpfs.FPFSDistort(modes_tmp=colnames)
    out = shapelet_shear.dm_dg(
        data2, ["fpfs_M00", "fpfs_M20", "fpfs_M22c", "fpfs_M22s"], 1
    )
    assert out.shape == (ndata, 4), "shear response has incorrect shape"

    res_00 = -np.sqrt(2.0) * data2[:, colnames.index("fpfs_M22c")]
    res_20 = -np.sqrt(6.0) * data2[:, colnames.index("fpfs_M42c")]
    res_22c = (
        1.0
        / np.sqrt(2.0)
        * (data2[:, colnames.index("fpfs_M00")] - data2[:, colnames.index("fpfs_M40")])
    )
    np.testing.assert_array_almost_equal(
        res_00, out[:, colnames.index("fpfs_M00")],
    )
    np.testing.assert_array_almost_equal(
        res_20, out[:, colnames.index("fpfs_M20")],
    )
    np.testing.assert_array_almost_equal(
        res_22c, out[:, colnames.index("fpfs_M22c")],
    )
    np.testing.assert_array_almost_equal(
        np.zeros(ndata), out[:, colnames.index("fpfs_M22s")],
    )
    return


def test_g2():
    print("testing for shapelets' g2 reponses")
    shapelet_shear = lpt.fpfs.FPFSDistort(modes_tmp=colnames)
    out = shapelet_shear.dm_dg(
        data2, ["fpfs_M00", "fpfs_M20", "fpfs_M22c", "fpfs_M22s"], 2
    )
    assert out.shape == (ndata, 4), "shear response has incorrect shape"

    res_00 = -np.sqrt(2.0) * data2[:, colnames.index("fpfs_M22s")]
    res_20 = -np.sqrt(6.0) * data2[:, colnames.index("fpfs_M42s")]
    res_22s = (
        1.0
        / np.sqrt(2.0)
        * (data2[:, colnames.index("fpfs_M00")] - data2[:, colnames.index("fpfs_M40")])
    )
    np.testing.assert_array_almost_equal(
        res_00, out[:, colnames.index("fpfs_M00")],
        )
    np.testing.assert_array_almost_equal(
        res_20, out[:, colnames.index("fpfs_M20")],
        )
    np.testing.assert_array_almost_equal(
        np.zeros(ndata), out[:, colnames.index("fpfs_M22c")],
        )
    np.testing.assert_array_almost_equal(res_22s,
        out[:, colnames.index("fpfs_M22s")],
        )
    return


if __name__ == "__main__":
    test_g1()
    test_g2()
