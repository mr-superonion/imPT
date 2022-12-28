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

test_fname = os.path.join(
    impt.fpfs.__data_dir__,
    "fpfs-cut32-0000-g1-0000.fits",
)

# impt.fpfs
cat = impt.fpfs.read_catalog(test_fname)
params = impt.fpfs.FpfsParams()
ell1 = impt.fpfs.FpfsE1(params)
ell2 = impt.fpfs.FpfsE2(params)
w_sel = impt.fpfs.FpfsWeightSelect(params)
w_det = impt.fpfs.FpfsWeightDetect(params)

# FPFS
data = fitsio.read(test_fname)
ndata = len(data)
ell_fpfs = fpfs.catalog.fpfsM2E(data, const=params.Const, noirev=False)
fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False, ratio=1.0)

selnm = []
cutsig = []
cut = []
selnm.append("detect2")
cutsig.append(params.sigma_v)
cut.append(params.lower_v)
selnm.append("M00")
cutsig.append(params.sigma_m00)
cut.append(params.lower_m00)
selnm.append("R2")
cutsig.append(params.sigma_r2)
cut.append(params.lower_r2)
selnm = np.array(selnm)
cutsig = np.array(cutsig)
cut = np.array(cut)

fs.clear_outcomes()
fs.update_selection_weight(selnm, cut, cutsig)
we1_fpfs = None
we2_fpfs = None
we1 = None


def test_measurement():
    print("testing measurement for FPFS's we1")
    we1 = w_sel * w_det * ell1
    np.testing.assert_array_almost_equal(
        we1.evaluate(cat),
        we1_fpfs,
    )

    print("testing measurement for FPFS's we2")
    we2 = w_sel * w_det * ell2
    np.testing.assert_array_almost_equal(
        we2.evaluate(cat),
        we2_fpfs,
    )
    return


def test_shear_response():
    print("testing measurement for FPFS's we2")
    esum = ell1 + ell2
    esum_f = ell_fpfs["fpfs_e1"] + ell_fpfs["fpfs_e2"]
    np.testing.assert_array_almost_equal(
        esum.evaluate(cat),
        esum_f,
    )

    print("testing shear response of FPFS's we1")
    dwe1_dg = impt.RespG1(we1)
    np.testing.assert_array_almost_equal(
        dwe1_dg.evaluate(cat),
        ell_fpfs["fpfs_R1E"],
    )

    print("testing shear response of FPFS's we1")
    dwe1_dg = impt.RespG1(we1)
    np.testing.assert_array_almost_equal(
        dwe1_dg.evaluate(cat),
        ell_fpfs["fpfs_R1E"],
    )
    return


if __name__ == "__main__":
    test_shear_response()
