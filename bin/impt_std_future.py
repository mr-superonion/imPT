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
import impt
import fitsio
import schwimmbad
import numpy as np
from impt.fpfs.future import prepare_func_e

from argparse import ArgumentParser
from configparser import ConfigParser
import logging

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("jax").setLevel(logging.CRITICAL)


def expand_2d_slice(point, dim1, dim2, bds, grid_size):
    # Validate input
    if len(point) != 5:
        raise ValueError("Input point must be 5D.")
    if dim1 not in range(5) or dim2 not in range(5):
        raise ValueError("Dimension indices must be in the range 0-4.")

    # Create a grid for the specified dimensions
    xrange = bds[dim1]
    yrange = bds[dim2]
    x = np.linspace(xrange[0], xrange[1], grid_size)
    y = np.linspace(yrange[0], yrange[1], grid_size)
    xv, yv = np.meshgrid(x, y)

    # Replicate the point in the 5D space
    replicated_point = np.tile(point, (grid_size, grid_size, 1))

    # Replace the dimensions with the grid values
    replicated_point[:, :, dim1] = xv
    replicated_point[:, :, dim2] = yv

    # Return the list of coordinates at the grids
    return replicated_point.reshape(-1, 5)


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

        # setup processor
        self.catdir = cparser.get("files", "cat_dir")
        ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
        self.cov_mat = fitsio.read(ncov_fname)
        self.ratio = ratio
        self.c0 = c0
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta
        return

    def get_range(self, icore):
        ibeg = self.min_id + icore * self.n_per_c
        iend = min(ibeg + self.n_per_c, self.max_id)
        id_range = list(range(ibeg, iend))
        if icore < len(self.rest_list):
            id_range.append(self.rest_list[icore])
        return id_range

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        if not os.path.isfile(in_nm):
            print("Cannot find input galaxy shear catalogs : %s " % (in_nm))
        mm = impt.fpfs.read_catalog(in_nm)
        # noise bias

        def fune(carry, ss):
            y = e1._obs_func(ss) - enoise._obs_func(ss)
            return carry + y, y

        def funr(carry, ss):
            y = res1._obs_func(ss) - rnoise._obs_func(ss)
            return carry + y, y

        e1_sum, _ = jax.lax.scan(fune, 0.0, mm)
        r1_sum, _ = jax.lax.scan(funr, 0.0, mm)
        del mm
        gc.collect()
        return e1_sum, r1_sum

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.empty((len(id_range), 2))
        # print("start core: %d, with id: %s" % (icore, id_range))
        for icount, ifield in enumerate(id_range):
            e1, enoise, res1, rnoise = prepare_func_e(
                cov_mat=self.cov_mat,
                ratio=self.ratio,
                c0=self.c0,
                c2=self.c2,
                alpha=self.alpha,
                beta=self.beta,
                g_comp=1,
            )
            in_nm1 = os.path.join(
                self.catdir,
                "src-%05d_g1-1_rot0.fits" % (ifield),
            )
            ell, e_r = self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)
            out[icount, 0] = ell
            out[icount, 1] = e_r
            del e1, enoise, res1, rnoise, ell, e_r
            gc.collect()
        return out


def process(args, pars):
    print("Current point: %s" % pars)
    params = {
        "ratio": pars[0],
        "c0": pars[1],
        "c2": pars[2],
        "alpha": pars[3],
        "beta": pars[4],
    }
    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores) as pool:
        ncores = get_processor_count(pool, args)
        if not isinstance(ncores, int):
            raise TypeError("ncores should be integer")
        core_list = np.arange(ncores)
        worker = Worker(
            args.config,
            min_id=args.min_id,
            max_id=args.max_id,
            ncores=ncores,
            **params,
        )
        outcome = np.vstack(list(pool.map(worker.run, core_list)))
        print(outcome.shape)
        std = np.std(outcome[:, 0]) / np.average(outcome[:, 1])
        print("std: %s" % std)
    return std


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs statistical error")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configure file name",
    )
    parser.add_argument(
        "--optimize",
        default="True",
        type=str,
        help="configure file name",
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
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

    cparser = ConfigParser()
    cparser.read(args.config)
    sum_dir = cparser.get("files", "sum_dir")

    if args.optimize == "True":
        from scipy.optimize import minimize

        bounds = [
            (0.5, 2.5),
            (2.0, 40.0),
            (2.0, 40.0),
            (0.2, 1.2),
            (0.2, 1.2),
        ]
        x0 = np.array([1.52, 2.46, 22.74, 0.35, 0.92])
        op = {"maxiter": 100, "disp": False, "xtol": 1e-1}
        res = minimize(
            process_opt,
            x0,
            bounds=bounds,
            method="Nelder-Mead",
        )
    if args.optimize == "False":
        bounds = [
            (1.3, 1.7),
            (1.0, 8.0),
            (10, 40),
            (0.45, 1.1),
            (0.0, 0.85),
        ]
        x0 = np.array([1.596, 2.46, 22.74, 0.83, 0.18])
        dim1, dim2 = 1, 2
        gsize = 15
        # dim1, dim2 = 3, 4
        # gsize = 20
        x_list = expand_2d_slice(x0, dim1, dim2, bounds, gsize)
        outcomes = np.stack(list(map(process_opt, x_list)))
        outcomes = np.vstack([x_list.T, outcomes])
        ofname = os.path.join(sum_dir, "%d_%d_lrange_g%d.fits" % (dim1, dim2, gsize))
        fitsio.write(ofname, outcomes)
    else:
        raise ValueError("optimize can only be 'True' or 'False'")
