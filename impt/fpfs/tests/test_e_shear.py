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
"""This unit test checks whether the FPFS nonlinear observables can do
arithmetic sum, subtract, multiply and divide correctly.
"""
import os
import fpfs
import fitsio
import numpy as np

import impt

wconst = 2.0
test_fname = os.path.join(
    impt.fpfs.__data_dir__,
    "fpfs-cut32-0000-g1-0000.fits",
)

data = fitsio.read(test_fname)
ndata = len(data)
ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=wconst, nn=None)

cat = impt.fpfs.read_catalog(test_fname)
params = impt.fpfs.FpfsParams(Const=wconst)

ell1 = impt.fpfs.FpfsE1(params)
ell2 = impt.fpfs.FpfsE2(params)


def test_add():
    print("testing measurement for FPFS's e1 + e2")
    esum = ell1 + ell2
    esum_f = ell_fpfs["fpfs_e1"] + ell_fpfs["fpfs_e2"]
    np.testing.assert_array_almost_equal(
        esum.evaluate(cat),
        esum_f,
    )

    print("testing shear response of FPFS's e1 + e1")
    te1 = ell1 + ell1
    dte1_dg = impt.RespG1(te1)
    np.testing.assert_array_almost_equal(
        dte1_dg.evaluate(cat),
        ell_fpfs["fpfs_R1E"] + ell_fpfs["fpfs_R1E"],
    )

    print("testing shear response of FPFS's e2 + e2")
    te2 = ell2 + ell2
    dte1_dg = impt.RespG2(te2)
    np.testing.assert_array_almost_equal(
        dte1_dg.evaluate(cat),
        ell_fpfs["fpfs_R2E"] + ell_fpfs["fpfs_R2E"],
    )
    return


def test_sub():
    print("testing measurement for FPFS's e1 - e2")
    ediff = ell1 - ell2
    ediff_f = ell_fpfs["fpfs_e1"] - ell_fpfs["fpfs_e2"]
    np.testing.assert_array_almost_equal(
        ediff.evaluate(cat),
        ediff_f,
    )
    return


def test_multiply():
    ratio = 21.2

    print("testing measurement for FPFS's e1 * %s" % ratio)
    esum = ell1 * ratio
    esum_f = ell_fpfs["fpfs_e1"] * ratio
    np.testing.assert_array_almost_equal(
        esum.evaluate(cat),
        esum_f,
    )

    print("testing shear response of FPFS's e1 * %s" % ratio)
    desum = impt.RespG1(esum)
    np.testing.assert_array_almost_equal(
        desum.evaluate(cat),
        ell_fpfs["fpfs_R1E"] * ratio,
    )

    print("testing measurement for FPFS's e2 * %s" % ratio)
    esum = ell2 * ratio
    esum_f = ell_fpfs["fpfs_e2"] * ratio
    np.testing.assert_array_almost_equal(
        esum.evaluate(cat),
        esum_f,
    )

    print("testing shear response of FPFS's e1 * %s" % ratio)
    desum = impt.RespG2(esum)
    np.testing.assert_array_almost_equal(
        desum.evaluate(cat),
        ell_fpfs["fpfs_R2E"] * ratio,
    )
    # print(desum.evaluate(cat))
    # print(ell_fpfs["fpfs_R2E"] * ratio)
    return


if __name__ == "__main__":
    test_add()
    test_sub()
    test_multiply()
