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
import jax.numpy as jnp

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
    print("testing selection on M00")
    params = impt.fpfs.FpfsParams(lower_m00=4.0, sigma_m00=0.5, lower_r2=-10.0)
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    e1_impt = impt.fpfs.FpfsE1(params)
    e2_impt = impt.fpfs.FpfsE2(params)
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["M00"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)
    print("     testing evaluation")
    we1 = e1_impt * w_sel
    we2 = e2_impt * w_sel
    np.testing.assert_array_almost_equal(
        fs.sumE1,
        jnp.sum(we1.evaluate(cat)),
    )
    np.testing.assert_array_almost_equal(
        fs.sumE2,
        jnp.sum(we2.evaluate(cat)),
    )
    print("     testing shear response")
    dwe1_dg1 = impt.RespG1(we1)
    dwe2_dg2 = impt.RespG2(we2)
    res_ad = jnp.sum(dwe1_dg1.evaluate(cat)) + jnp.sum(dwe2_dg2.evaluate(cat))
    res_fpfs = fs.corR1 + fs.sumR1 + fs.corR2 + fs.sumR2
    np.testing.assert_array_almost_equal(
        res_ad,
        res_fpfs,
    )
    return


def test_R2():
    print("testing selection weight on R2")
    params = impt.fpfs.FpfsParams(
        lower_m00=-4.0, sigma_m00=0.5, lower_r2=0.12, sigma_r2=0.2
    )
    e1_impt = impt.fpfs.FpfsE1(params)
    e2_impt = impt.fpfs.FpfsE2(params)
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["R2"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)
    print("     testing evaluation")
    we1 = e1_impt * w_sel
    we2 = e2_impt * w_sel
    np.testing.assert_array_almost_equal(
        fs.sumE1,
        jnp.sum(we1.evaluate(cat)),
    )
    np.testing.assert_array_almost_equal(
        fs.sumE2,
        jnp.sum(we2.evaluate(cat)),
    )
    print("     testing shear response")
    dwe1_dg1 = impt.RespG1(we1)
    dwe2_dg2 = impt.RespG2(we2)
    res_ad = jnp.sum(dwe1_dg1.evaluate(cat)) + jnp.sum(dwe2_dg2.evaluate(cat))
    res_fpfs = fs.corR1 + fs.sumR1 + fs.corR2 + fs.sumR2
    np.testing.assert_array_almost_equal(
        res_ad,
        res_fpfs,
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
    e1_impt = impt.fpfs.FpfsE1(params)
    e2_impt = impt.fpfs.FpfsE2(params)
    w_det = impt.fpfs.FpfsWeightDetect(params)
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["detect2"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)
    print("     testing evaluation")
    we1 = e1_impt * w_det
    we2 = e2_impt * w_det
    np.testing.assert_array_almost_equal(
        fs.sumE1,
        jnp.sum(we1.evaluate(cat)),
    )
    np.testing.assert_array_almost_equal(
        fs.sumE2,
        jnp.sum(we2.evaluate(cat)),
    )
    print("     testing shear response")
    dwe1_dg1 = impt.RespG1(we1)
    dwe2_dg2 = impt.RespG2(we2)
    res_ad = jnp.sum(dwe1_dg1.evaluate(cat)) + jnp.sum(dwe2_dg2.evaluate(cat))
    res_fpfs = fs.corR1 + fs.sumR1 + fs.corR2 + fs.sumR2
    np.testing.assert_array_almost_equal(
        res_ad,
        res_fpfs,
    )
    return


def test_all():
    print("testing final selection weight")
    # evaluate impt.fpfs
    params = impt.fpfs.FpfsParams(
        lower_m00=3.5,
        sigma_m00=0.5,
        lower_r2=0.12,
        sigma_r2=0.2,
        sigma_v=0.2,
    )
    e1_impt = impt.fpfs.FpfsE1(params)
    e2_impt = impt.fpfs.FpfsE2(params)
    w_det = impt.fpfs.FpfsWeightDetect(params)
    w_sel = impt.fpfs.FpfsWeightSelect(params)
    we1 = e1_impt * w_det * w_sel
    we2 = e2_impt * w_det * w_sel

    # evaluate fpfs
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["detect2", "M00", "R2"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)

    np.testing.assert_array_almost_equal(
        fs.sumE1,
        jnp.sum(we1.evaluate(cat)),
    )
    np.testing.assert_array_almost_equal(
        fs.sumE2,
        jnp.sum(we2.evaluate(cat)),
    )
    print("     testing shear response")
    dwe1_dg1 = impt.RespG1(we1)
    dwe2_dg2 = impt.RespG2(we2)
    res_ad = jnp.sum(dwe1_dg1.evaluate(cat)) + jnp.sum(dwe2_dg2.evaluate(cat))
    res_fpfs = fs.corR1 + fs.sumR1 + fs.corR2 + fs.sumR2
    np.testing.assert_array_almost_equal(
        res_ad,
        res_fpfs,
    )
    return


def test_all2():
    print("testing final selection weight")
    # evaluate impt.fpfs
    params = impt.fpfs.FpfsParams(
        lower_m00=3.5,
        lower_r2=0.03,
        lower_v=0.3,
        upper_r2=1000.0,
        sigma_m00=2.0,
        sigma_r2=2.0,
        sigma_v=1.5,
    )
    we1 = impt.fpfs.FpfsWeightE1(params)
    we2 = impt.fpfs.FpfsWeightE2(params)

    # evaluate fpfs
    ell_fpfs = fpfs.catalog.fpfs_m2e(data, const=params.Const, nn=None)
    fs = fpfs.catalog.summary_stats(data, ell_fpfs, use_sig=False)
    selnm = np.array(["detect", "M00", "R2"])
    fs = impt.fpfs.test_utils.initialize_FPFS(fs, selnm, params)

    np.testing.assert_array_almost_equal(
        fs.sumE1,
        jnp.sum(we1.evaluate(cat)),
    )
    np.testing.assert_array_almost_equal(
        fs.sumE2,
        jnp.sum(we2.evaluate(cat)),
    )
    print("     testing shear response")
    dwe1_dg1 = impt.RespG1(we1)
    dwe2_dg2 = impt.RespG2(we2)
    res_ad = jnp.sum(dwe1_dg1.evaluate(cat)) + jnp.sum(dwe2_dg2.evaluate(cat))
    res_fpfs = fs.corR1 + fs.sumR1 + fs.corR2 + fs.sumR2
    np.testing.assert_array_almost_equal(
        res_ad,
        res_fpfs,
    )
    return


if __name__ == "__main__":
    test_flux()
    test_R2()
    test_peak()
    test_all()
    test_all2()
