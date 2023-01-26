# impt autodiff pipeline
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
"""This unit test checks whether FPFS's linear observables (shapelets) shear
response is implemented correctly. """

import os
import fitsio
import numpy as np

import impt
import impt.fpfs.default as df
from impt.fpfs.default import indexes as did

test_fname = os.path.join(
    impt.fpfs.__data_dir__,
    "fpfs-cut32-0000-g1-0000.fits",
)

data = fitsio.read(test_fname)[df.col_names]
data2 = impt.fpfs.read_catalog(test_fname)
ndata = len(data)


def test_g1():
    print("testing for shapelets' g1 reponses")
    linres = impt.fpfs.FpfsLinResponse()
    out = linres.dg1(data2)
    assert out.shape == (ndata, df.ncol), "shear response has incorrect shape"

    res_00 = -np.sqrt(2.0) * data2[:, did["m22c"]]
    res_20 = -np.sqrt(6.0) * data2[:, did["m42c"]]
    res_22c = 1.0 / np.sqrt(2.0) * (data2[:, did["m00"]] - data2[:, did["m40"]])
    np.testing.assert_array_almost_equal(
        res_00,
        out[:, did["m00"]],
    )
    np.testing.assert_array_almost_equal(
        res_20,
        out[:, did["m20"]],
    )
    np.testing.assert_array_almost_equal(
        res_22c,
        out[:, did["m22c"]],
    )
    np.testing.assert_array_almost_equal(
        np.zeros(ndata),
        out[:, did["m22s"]],
    )
    return


def test_g2():
    print("testing for shapelets' g2 reponses")
    linres = impt.fpfs.FpfsLinResponse()
    out = linres.dg2(data2)
    assert out.shape == (ndata, df.ncol), "shear response has incorrect shape"

    res_00 = -np.sqrt(2.0) * data2[:, did["m22s"]]
    res_20 = -np.sqrt(6.0) * data2[:, did["m42s"]]
    res_22s = 1.0 / np.sqrt(2.0) * (data2[:, did["m00"]] - data2[:, did["m40"]])
    np.testing.assert_array_almost_equal(
        res_00,
        out[:, did["m00"]],
    )
    np.testing.assert_array_almost_equal(
        res_20,
        out[:, did["m20"]],
    )
    np.testing.assert_array_almost_equal(
        np.zeros(ndata),
        out[:, did["m22c"]],
    )
    np.testing.assert_array_almost_equal(
        res_22s,
        out[:, did["m22s"]],
    )
    return


if __name__ == "__main__":
    test_g1()
    test_g2()
