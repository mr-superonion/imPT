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


def test_flux():
    print("testing selection weight on M00")
    params = impt.fpfs.FpfsParams(lower_m00=4.0, sigma_m00=0.5, lower_r2=-10.0)
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["M00"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)
    np.testing.assert_array_almost_equal(
        fs.ws,
        w_sel.evaluate(cat),
    )
    return


def test_R2():
    print("testing selection weight on R2")
    params = impt.fpfs.FpfsParams(
        lower_m00=-4.0, sigma_m00=0.5, lower_r2=0.12, sigma_r2=0.2
    )
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["R2"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)
    np.testing.assert_array_almost_equal(
        fs.ws,
        w_sel.evaluate(cat),
    )
    return


def test_peak():
    print("testing selection weight on peak modes")
    params = impt.fpfs.FpfsParams(
        lower_m00=-4.0,
        sigma_m00=0.5,
        lower_r2=-4.0,
        sigma_r2=0.2,
        sigma_v=0.2,
    )
    w_det = impt.fpfs.FpfsWeightDetect(params)
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["detect2"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)
    np.testing.assert_array_almost_equal(
        fs.ws,
        w_det.evaluate(cat),
    )
    return


if __name__ == "__main__":
    test_flux()
    test_R2()
    test_peak()
