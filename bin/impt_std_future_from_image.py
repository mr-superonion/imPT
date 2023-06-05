#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20220312 Xiangchong Li.
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
import os
import gc
import jax
import fpfs
import pickle
import schwimmbad
import numpy as np
from scipy.optimize import minimize
from impt.fpfs.future import prepare_func_e
from impt.fpfs.default import indexes as did

import lsst.geom as lsstgeom
import lsst.afw.image as afwimage
from descwl_shear_sims.psfs import make_dm_psf

from argparse import ArgumentParser
from configparser import ConfigParser
import logging

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("jax").setLevel(logging.CRITICAL)


def get_processor_count(pool, args):
    if isinstance(pool, schwimmbad.MPIPool):
        # MPIPool
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() - 1
    elif isinstance(pool, schwimmbad.MultiPool):
        # MultiPool
        return args.n_cores
    else:
        # SerialPool
        return 1


band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
    "a": 4,
}

nstd_map = {
    "g": 0.315,
    "r": 0.371,
    "i": 0.595,
    "z": 1.155,
    "a": 0.2793,
}

w_map = {
    "g": 0.12503653,
    "r": 0.47022727,
    "i": 0.30897575,
    "z": 0.09576044,
}


def get_seed_from_fname(fname, band):
    fid = int(fname.split("image-")[-1].split("_")[0]) + 212
    rid = int(fname.split("rot")[1][0])
    bid = band_map[band]
    return (fid * 2 + rid) * 4 + bid


class Worker(object):
    def __init__(
        self,
        config_name,
        min_id=0,
        max_id=1000,
        ncores=1,
        ratio=1.3,
        c0=4.0,
        c2=4.0,
        alpha=0.2,
        beta=0.8,
        sigma_mea=0.6,
        sigma_det=0.5,
    ):
        cparser = ConfigParser()
        cparser.read(config_name)

        # simulation parameter
        nids = max_id - min_id
        self.n_per_c = nids // ncores
        self.mid = nids % ncores
        self.min_id = min_id
        self.max_id = max_id
        self.rest_list = list(np.arange(ncores * self.n_per_c, nids) + min_id)
        print("number of files per core is: %d" % self.n_per_c)
        # print(self.min_id, self.max_id)

        self.img_dir = cparser.get("procsim", "img_dir")
        self.rcut = cparser.getint("FPFS", "rcut")
        self.nnord = cparser.getint("FPFS", "nnord", fallback=4)
        ngrid = 2 * self.rcut
        self.psf_fname = cparser.get("procsim", "psf_fname")
        self.band = cparser.get("survey", "band")
        self.nstd_f = nstd_map[self.band]
        self.noise_pow = np.ones((ngrid, ngrid)) * self.nstd_f**2.0 * ngrid**2.0
        self.scale = 0.2

        # setup processor
        self.ratio = ratio
        self.c0 = c0
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta
        self.sigma_mea = sigma_mea
        self.sigma_det = sigma_det
        return

    def get_range(self, icore):
        ibeg = self.min_id + icore * self.n_per_c
        iend = min(ibeg + self.n_per_c, self.max_id)
        id_range = list(range(ibeg, iend))
        if icore < len(self.rest_list):
            id_range.append(self.rest_list[icore])
        return id_range

    def prepare_psf(self, exposure, rcut, ngrid2):
        # pad to (64, 64) and then cut off
        ngrid = 64
        beg = ngrid // 2 - rcut
        end = beg + 2 * rcut
        bbox = exposure.getBBox()
        bcent = bbox.getCenter()
        psf_model = exposure.getPsf()
        psf_array = psf_model.computeImage(lsstgeom.Point2I(bcent)).getArray()
        npad = (ngrid - psf_array.shape[0]) // 2
        psf_array2 = np.pad(psf_array, (npad + 1, npad), mode="constant")[
            beg:end, beg:end
        ]
        del npad
        # pad to exposure size
        npad = (ngrid2 - psf_array.shape[0]) // 2
        psf_array3 = np.pad(psf_array, (npad + 1, npad), mode="constant")
        return psf_array2, psf_array3

    def prepare_image(self, icore):
        fname = os.path.join(
            self.img_dir,
            "image-%05d_g1-0_rot0_%s.fits" % (icore, self.band),
        )
        exposure = afwimage.ExposureF.readFits(fname)
        # PSF
        with open(self.psf_fname, "rb") as f:
            psf_dict = pickle.load(f)
        psf_dm = make_dm_psf(**psf_dict)
        exposure.setPsf(psf_dm)
        del psf_dm
        self.image_nx = exposure.getWidth()
        psf_array2, psf_array3 = self.prepare_psf(
            exposure,
            self.rcut,
            self.image_nx,
        )
        # noise covariance
        noise_task = fpfs.image.measure_noise_cov(
            psf_array2,
            sigma_arcsec=self.sigma_mea,
            nnord=self.nnord,
            pix_scale=self.scale,
            sigma_detect=self.sigma_det,
        )
        cov_mat = np.array(noise_task.measure(self.noise_pow))
        del noise_task
        # image
        if False:
            seed = get_seed_from_fname(fname, self.band)
            rng = np.random.RandomState(seed)
            gal_array = exposure.getMaskedImage().getImage().getArray() + rng.normal(
                scale=self.nstd_f,
                size=(self.image_nx, self.image_nx),
            )
            del rng
        else:
            gal_array = exposure.getMaskedImage().getImage().getArray()
        # print(np.sqrt(np.diag(cov_mat)))
        del exposure
        gc.collect()
        return gal_array, psf_array2, psf_array3, cov_mat

    def process_image(self, icore):
        gal_array, psf_array2, psf_array3, cov_mat = self.prepare_image(icore)
        # measurement task
        meas_task = fpfs.image.measure_source(
            psf_array2,
            sigma_arcsec=self.sigma_mea,
            nnord=self.nnord,
            pix_scale=self.scale,
            sigma_detect=self.sigma_det,
        )

        std_modes = np.sqrt(np.diagonal(cov_mat))
        idm00 = fpfs.catalog.indexes["m00"]
        idv0 = fpfs.catalog.indexes["v0"]
        thres = 10.0 * std_modes[idm00] * self.scale**2.0
        thres2 = -1.5 * std_modes[idv0] * self.scale**2.0

        coords = fpfs.image.detect_sources(
            gal_array,
            psf_array3,
            sigmaf=meas_task.sigmaf,
            sigmaf_det=meas_task.sigmaf_det,
            thres=thres,
            thres2=thres2,
            bound=self.rcut + 5,
        )
        mm = meas_task.measure(gal_array, coords)
        del meas_task
        msk = (mm[:, did["m00"]] + mm[:, did["m20"]]) > 1e-5
        mm = mm[msk]
        del msk
        gc.collect()
        return mm, cov_mat

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.empty(len(id_range))
        # print("start core: %d, with id: %s" % (icore, id_range))
        for icount, ifield in enumerate(id_range):
            mm, cov_mat = self.process_image(ifield)
            e1, enoise, res1, rnoise = prepare_func_e(
                cov_mat=cov_mat,
                ratio=self.ratio,
                c0=self.c0,
                c2=self.c2,
                alpha=self.alpha,
                beta=self.beta,
                g_comp=1,
            )
            del cov_mat

            def fune(carry, ss):
                y = e1._obs_func(ss) - enoise._obs_func(ss)  # noqa
                return carry + y, None

            def funr(carry, ss):
                y = res1._obs_func(ss) - rnoise._obs_func(ss)  # noqa
                return carry + y, None

            e_sum, _ = jax.lax.scan(fune, 0.0, mm)
            r_sum, _ = jax.lax.scan(funr, 0.0, mm)
            out[icount] = float(e_sum / r_sum)
            del e1, enoise, res1, rnoise, mm
            gc.collect()
        return out


def process(args, pars):
    params = {
        "ratio": pars[0],
        "c0": pars[1],
        "c2": pars[2],
        "alpha": pars[3],
        "beta": pars[4],
        "sigma_mea": pars[5],
        "sigma_det": pars[6],
    }
    print("Current point: %s" % pars)
    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores) as pool:
        ncores = get_processor_count(pool, args)
        assert isinstance(ncores, int)
        core_list = np.arange(ncores)
        worker = Worker(
            args.config,
            min_id=args.min_id,
            max_id=args.max_id,
            ncores=ncores,
            **params,
        )
        outcome = np.hstack(list(pool.map(worker.run, core_list)))
        print(len(outcome))
        std = np.std(outcome)
        print("std: %s" % std)
    gc.collect()
    return std


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configure file name",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=100,
        type=int,
        help="id number, e.g. 1000",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    args = parser.parse_args()
    process_opt = lambda _: process(args=args, pars=_)
    bounds = [
        (0.5, 2.5),
        (1.5, 40.0),
        (1.5, 40.0),
        (0.2, 1.2),
        (0.2, 1.2),
        (0.45, 0.75),
        (0.45, 0.75),
    ]
    x0 = np.array([1.8, 2.548, 22.774, 0.366, 0.960, 0.536, 0.507])
    op = {"maxiter": 400, "disp": False, "xtol": 1e-1}
    res = minimize(
        process_opt,
        x0,
        bounds=bounds,
        method="Nelder-Mead",
        # method="Powell",
    )
    print(res)
