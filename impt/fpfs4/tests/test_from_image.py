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
import fpfs
import impt
import galsim
import numpy as np
import jax.numpy as jnp

test_thres = 1e-4
params = impt.fpfs4.FpfsParams(
    lower_m00=0.0,
    lower_r2=0.0,
    lower_v=-10.0,  # cannot test detection with 4 galaxies; need 6 galaxies
    sigma_m00=0.4,
    sigma_r2=0.8,
    sigma_v=0.2,
)


def simulate_gal_psf(scale, ind0, rcut):
    out_dir = "galaxy_basicCenter_psf60"
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=64, ny=64, scale=scale)
        .array
    )
    psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]
    gal_data = fpfs.simutil.make_basic_sim(
        out_dir,
        psf_obj=psf_obj,
        gname="g1-0000",
        ind0=ind0,
        ny=64,
        nx=256,
        scale=scale,
        do_write=False,
        return_array=True,
    )

    # force detection at center
    indx = np.arange(32, 256, 64)
    indy = np.arange(32, 64, 64)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    return gal_data, psf_data, coords


def do_test(scale, ind0, rcut):
    gal_data, psf_data, coords = simulate_gal_psf(scale, ind0, rcut)
    # test shear estimation
    fpfs_task = fpfs.image.measure_source(psf_data, sigma_arcsec=0.5, nnord=6)
    # linear observables
    # detection
    p1 = 32 - rcut
    p2 = 64 * 2 - rcut
    psf_data2 = jnp.pad(psf_data, ((p1, p1), (p2, p2)))
    coords2 = fpfs.image.detect_sources(
        gal_data,
        psf_data2,
        sigmaf_det=fpfs_task.sigmaf_det,
        sigmaf=fpfs_task.sigmaf_det,
        thres=0.01,
        thres2=0.00,
    )
    assert np.all(coords2 == coords)
    cat = fpfs_task.measure(gal_data, coords2)

    e1 = impt.fpfs4.FpfsWeightE41(params)
    dg1 = impt.RespG1(e1)
    shear = jnp.sum(e1.evaluate(cat)) / jnp.sum(dg1.evaluate(cat))
    print(shear)
    assert np.all(np.abs(shear + 0.02) < test_thres)
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 2, 16)
    do_test(0.168, 6, 32)
    return


if __name__ == "__main__":
    test_hsc()
