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
"""This unit test checks whether lenspt can recover the first and second
version of FPFS shear estimator (without considering detection bias and
selection bias)
"""
import fpfs
import fitsio
import numpy as np
import lenspt as lpt


colnames = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
]
wconst = 2.0
data = fitsio.read("data/fpfs-cut32-0000-g1-0000.fits")
ndata = len(data)
cov_data = lpt.fpfsCov2lptCov(data, colnames)

ell_fpfs = fpfs.catalog.fpfsM2E(data, const=wconst, noirev=False)
ell_fpfs_corr = fpfs.catalog.fpfsM2E(data, const=wconst, noirev=True)
noicorr_fpfs_e1 = ell_fpfs["fpfs_e1"] - ell_fpfs_corr["fpfs_e1"]
noicorr_fpfs_e2 = ell_fpfs["fpfs_e2"] - ell_fpfs_corr["fpfs_e2"]

cat = lpt.Catalog(
    data="data/fpfs-cut32-0000-g1-0000.fits",
    mode_names=colnames,
    noise_cov=cov_data,
)


def test_add():
    print("testing measurement for FPFS's e1 + e2")
    ell1 = lpt.fpfs.WeightedE1(wconst=wconst)
    ell2 = lpt.fpfs.WeightedE2(wconst=wconst)
    esum = ell1 + ell2
    esum_f = ell_fpfs["fpfs_e1"] + ell_fpfs["fpfs_e2"]
    np.testing.assert_array_almost_equal(esum.evaluate(cat), esum_f)
    print("testing shear response of FPFS's e1 + e1")
    te1 = ell1 + ell1
    dte1_dg = lpt.G1Perturb1(te1)
    np.testing.assert_array_almost_equal(
        dte1_dg.evaluate(cat),
        ell_fpfs["fpfs_R1E"] +ell_fpfs["fpfs_R1E"],
    )
    return



if __name__ == "__main__":
    test_add()
