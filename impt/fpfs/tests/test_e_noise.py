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
"""This unit test checks whether the noise bias estimation by the autodiff is
consistent with FPFS analytic results on e_fpfs (ellipticity) and R_fpfs (shear
response)
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

# FPFS
# read data with fitsio
data = fitsio.read(test_fname)
ndata = len(data)

ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=wconst, nn=None)
ell_fpfs_corr = fpfs.catalog.fpfs_m2e(data, const=wconst, nn=data)
noicorr_fpfs_e1 = ell_fpfs["fpfs_e1"] - ell_fpfs_corr["fpfs_e1"]
noicorr_fpfs_e2 = ell_fpfs["fpfs_e2"] - ell_fpfs_corr["fpfs_e2"]
noicorr_fpfs_de1dg1 = ell_fpfs["fpfs_R1E"] - ell_fpfs_corr["fpfs_R1E"]
noicorr_fpfs_de2dg2 = ell_fpfs["fpfs_R2E"] - ell_fpfs_corr["fpfs_R2E"]

# impt.fpfs
params = impt.fpfs.FpfsParams(Const=wconst)
cat = impt.fpfs.read_catalog(test_fname)
noise_cov = impt.fpfs.utils.fpfscov_to_imptcov(data)

ell1 = impt.fpfs.FpfsE1(params)
ell2 = impt.fpfs.FpfsE2(params)

ell1_dg1 = impt.RespG1(ell1)
ell2_dg2 = impt.RespG2(ell2)


def test_e1e2():
    print("testing noise bias correction for FPFS's e1")
    bnoise = impt.BiasNoise(ell1, noise_cov)
    np.testing.assert_array_almost_equal(
        bnoise.evaluate(cat),
        noicorr_fpfs_e1,
    )

    print("testing noise bias correction for FPFS's e2")
    bnoise = impt.BiasNoise(ell2, noise_cov)
    np.testing.assert_array_almost_equal(
        bnoise.evaluate(cat),
        noicorr_fpfs_e2,
    )

    print("testing noise bias correction for FPFS's RE1")
    bnoise = impt.BiasNoise(ell1_dg1, noise_cov)
    np.testing.assert_array_almost_equal(
        bnoise.evaluate(cat),
        noicorr_fpfs_de1dg1,
    )

    print("testing noise bias correction for FPFS's RE2")
    bnoise = impt.BiasNoise(ell2_dg2, noise_cov)
    np.testing.assert_array_almost_equal(
        bnoise.evaluate(cat),
        noicorr_fpfs_de2dg2,
    )
    print(np.sum(bnoise.evaluate(cat)))
    print(np.sum(noicorr_fpfs_de2dg2))
    return


if __name__ == "__main__":
    test_e1e2()
