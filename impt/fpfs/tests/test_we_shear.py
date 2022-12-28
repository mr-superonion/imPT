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
# FPFS
data = fitsio.read(test_fname)


def initialize_FPFS(fs, snlist, params):
    cutsig = []
    cut = []
    for sn in snlist:
        if sn == "detect2":
            cutsig.append(params.sigma_v)
            cut.append(params.lower_v)
        elif sn == "M00":
            cutsig.append(params.sigma_m00)
            cut.append(params.lower_m00)
        elif sn == "R2":
            cutsig.append(params.sigma_r2)
            cut.append(params.lower_r2)
    cutsig = np.array(cutsig)
    cut = np.array(cut)
    fs.clear_outcomes()
    fs.update_selection_weight(snlist, cut, cutsig)
    return fs


def test_weights():
    print("testing selection weight on M00")
    params = impt.fpfs.FpfsParams(lower_m00=4.0, sigma_m00=0.5, lower_r2=-10.0)
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    ell_fpfs = fpfs.catalog.fpfsM2E(data, const=params.Const, noirev=False)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False, ratio=1.0)
    selnm = np.array(["M00"])
    fs = initialize_FPFS(fs, selnm, params)
    np.testing.assert_array_almost_equal(
        fs.ws,
        w_sel.evaluate(cat),
    )

    print("testing selection weight on R2")
    params = impt.fpfs.FpfsParams(
        lower_m00=-4.0, sigma_m00=0.5, lower_r2=0.12, sigma_r2=0.2
    )
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    ell_fpfs = fpfs.catalog.fpfsM2E(data, const=params.Const, noirev=False)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False, ratio=1.0)
    selnm = np.array(["R2"])
    fs = initialize_FPFS(fs, selnm, params)
    np.testing.assert_array_almost_equal(
        fs.ws,
        w_sel.evaluate(cat),
    )

    print("testing selection weight on peak modes")
    params = impt.fpfs.FpfsParams(
        lower_m00=-4.0,
        sigma_m00=0.5,
        lower_r2=-4.0,
        sigma_r2=0.2,
        sigma_v=0.2,
    )
    w_det = impt.fpfs.FpfsWeightDetect(params)
    ell_fpfs = fpfs.catalog.fpfsM2E(data, const=params.Const, noirev=False)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False, ratio=1.0)
    selnm = np.array(["detect2"])
    fs = initialize_FPFS(fs, selnm, params)
    np.testing.assert_array_almost_equal(
        fs.ws,
        w_det.evaluate(cat),
    )
    return


if __name__ == "__main__":
    test_weights()
