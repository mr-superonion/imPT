# lensPT autodiff pipeline
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
import fpfs
import fitsio
import numpy as np
import lensPT as lpt


ndata = 30
data = fitsio.read("./fpfs-cut32-0000-g1-0000.fits")
ell_fpfs = fpfs.catalog.fpfsM2E(data, const=1.0, noirev=False)
colnames = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
]
cat = lpt.observable.Catalog("./fpfs-cut32-0000-g1-0000.fits", mode_names=colnames)


def test_e1():
    print("testing for FPFS's e1")
    ell1 = lpt.fpfs.weighted_e1(Const=1.0)
    np.testing.assert_array_almost_equal(ell1.evaluate(cat), ell_fpfs["fpfs_e1"])
    return


def test_e2():
    print("testing for FPFS's e2")
    ell2 = lpt.fpfs.weighted_e2(Const=1.0)
    np.testing.assert_array_almost_equal(ell2.evaluate(cat), ell_fpfs["fpfs_e2"])
    return


if __name__ == "__main__":
    test_e1()
    test_e2()
