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
"""This unit test checks whether lensPT can recover the first and second
version of FPFS shear estimator (without considering detection bias and
selection bias)
"""
import fpfs
import fitsio
import numpy as np
import lensPT as lpt


colnames = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
]
Const = 2.0
data = fitsio.read("data/fpfs-cut32-0000-g1-0000.fits")
ndata = len(data)
cov_data=lpt.fpfsCov2lptCov(data, colnames)

ell_fpfs = fpfs.catalog.fpfsM2E(data, const=Const, noirev=False)
ell_fpfs_corr = fpfs.catalog.fpfsM2E(data, const=Const, noirev=True)
noicorr_fpfs_e1 = ell_fpfs["fpfs_e1"] - ell_fpfs_corr["fpfs_e1"]
noicorr_fpfs_e2 = ell_fpfs["fpfs_e2"] - ell_fpfs_corr["fpfs_e2"]

cat = lpt.Catalog(
    data="data/fpfs-cut32-0000-g1-0000.fits",
    mode_names=colnames,
    noise_cov=cov_data,
)


def test_e1():
    print("testing measurement for FPFS's e1")
    ell1 = lpt.fpfs.weightedE1(Const=Const)
    np.testing.assert_array_almost_equal(ell1.evaluate(cat), ell_fpfs["fpfs_e1"])
    print("testing shear response of FPFS's e1")
    de1_dg = lpt.g1Perturb1(ell1)
    np.testing.assert_array_almost_equal(
        de1_dg.evaluate(cat),
        ell_fpfs["fpfs_R1E"],
    )
    print("testing noise response of FPFS's e1")
    noiseE1=lpt.noisePerturb2(ell1)
    np.testing.assert_array_almost_equal(noiseE1.evaluate(cat), noicorr_fpfs_e1)
    print("testing noise response of FPFS's Re1")
    noiseR1=lpt.noisePerturb2(de1_dg)
    np.testing.assert_array_almost_equal(
            noiseR1.evaluate(cat),
            ell_fpfs['fpfs_R1E'] - ell_fpfs_corr['fpfs_R1E'],
            )
    return


def test_e2():
    print("testing measurement for FPFS's e2")
    ell2 = lpt.fpfs.weightedE2(Const=Const)
    np.testing.assert_array_almost_equal(ell2.evaluate(cat), ell_fpfs["fpfs_e2"])
    print("testing shear response of FPFS's e2")
    de2_dg = lpt.g2Perturb1(ell2)
    np.testing.assert_array_almost_equal(
        de2_dg.evaluate(cat),
        ell_fpfs["fpfs_R2E"],
    )
    print("testing noise response of FPFS's e2")
    noiseE2=lpt.noisePerturb2(ell2)
    np.testing.assert_array_almost_equal(noiseE2.evaluate(cat), noicorr_fpfs_e2)
    print("testing noise response of FPFS's Re2")
    noiseR2=lpt.noisePerturb2(de2_dg)
    np.testing.assert_array_almost_equal(
            noiseR2.evaluate(cat),
            ell_fpfs['fpfs_R2E'] - ell_fpfs_corr['fpfs_R2E'],
            )
    return


if __name__ == "__main__":
    test_e1()
    test_e2()
