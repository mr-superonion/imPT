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
import gc
import os
import jax
import impt
import fitsio
import schwimmbad
import numpy as np
import jax.numpy as jnp
from functools import partial

from argparse import ArgumentParser
from configparser import ConfigParser

os.environ["JAX_PLATFORM_NAME"] = "cpu"


class Worker(object):
    def __init__(self, config_name, gver="g1"):
        cparser = ConfigParser()
        cparser.read(config_name)
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")
        cov_fname = os.path.join(
            impt.fpfs.__data_dir__, "modes_cov_mat_paper3_045.fits"
        )
        self.cov_mat = jnp.array(fitsio.read(cov_fname))

        # setup processor
        self.indir = cparser.get("procsim", "input_dir")
        self.outdir = cparser.get("procsim", "output_dir")
        os.makedirs(self.outdir, exist_ok=True)
        self.simname = cparser.get("procsim", "sim_name")

        self.rcut = cparser.getint("FPFS", "rcut")

        # This task change the cut on one observable and see how the biases changes.
        # Here is  the observable used for test
        self.upper_mag = cparser.getfloat("FPFS", "cut_mag")
        self.lower_m00 = 10 ** ((self.magz - self.upper_mag) / 2.5)

        if not os.path.exists(self.indir):
            raise FileNotFoundError("Cannot find input directory: %s!" % self.indir)
        print("The input directory for galaxy shear catalogs is %s. " % self.indir)
        # setup WL distortion parameter
        self.gver = gver
        return

    @partial(jax.jit, static_argnums=(0,))
    def measure(self, data):
        params = impt.fpfs.FpfsParams(
            Const=20,
            lower_m00=self.lower_m00,
            sigma_m00=0.2,
            lower_r2=0.05,
            upper_r2=2.0,
            sigma_r2=0.2,
            sigma_v=0.2,
        )
        funcnm = "ts2"
        e1_impt = impt.fpfs.FpfsE1(params, func_name=funcnm)
        w_det = impt.fpfs.FpfsWeightDetect(params, func_name=funcnm)
        w_sel = impt.fpfs.FpfsWeightSelect(params, func_name=funcnm)

        # ellipticity
        e1 = e1_impt * w_sel * w_det
        enoise = impt.BiasNoise(e1, self.cov_mat)
        e1_sum = jnp.sum(e1.evaluate(data))
        e1_sum = e1_sum - jnp.sum(enoise.evaluate(data))
        del enoise
        gc.collect()

        # shear response
        res1 = impt.RespG1(e1)
        rnoise = impt.BiasNoise(res1, self.cov_mat)
        r1_sum = jnp.sum(res1.evaluate(data))
        r1_sum = r1_sum - jnp.sum(rnoise.evaluate(data))
        del res1, rnoise, e1
        gc.collect()
        return e1_sum, r1_sum

    def run(self, ind0):
        out_nm = os.path.join(self.outdir, "%04d.fits" % ind0)
        if os.path.isfile(out_nm):
            return
        pp = "cut%d" % self.rcut
        in_nm1 = os.path.join(
            self.indir, "fpfs-%s-%04d-%s-0000.fits" % (pp, ind0, self.gver)
        )
        in_nm2 = os.path.join(
            self.indir, "fpfs-%s-%04d-%s-2222.fits" % (pp, ind0, self.gver)
        )
        assert os.path.isfile(in_nm1) & os.path.isfile(
            in_nm2
        ), "Cannot find input galaxy shear catalogs : %s , %s" % (in_nm1, in_nm2)
        mm1 = impt.fpfs.read_catalog(in_nm1)
        mm2 = impt.fpfs.read_catalog(in_nm2)

        # names= [('cut','<f8'), ('de','<f8'), ('eA','<f8')
        # ('res','<f8')]
        out = np.zeros((4, 1))
        sum_e1_1, sum_r1_1 = self.measure(mm1)
        sum_e1_2, sum_r1_2 = self.measure(mm2)
        del mm1, mm2
        gc.collect()
        out[0, 0] = self.upper_mag
        out[1, 0] = sum_e1_2 - sum_e1_1
        out[2, 0] = (sum_e1_1 + sum_e1_2) / 2.0
        out[3, 0] = (sum_r1_1 + sum_r1_2) / 2.0
        fitsio.write(out_nm, out)
        return out


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--minId", required=True, type=int, help="minimum id number, e.g. 0"
    )
    parser.add_argument(
        "--maxId", required=True, type=int, help="maximum id number, e.g. 4000"
    )
    parser.add_argument("--config", required=True, type=str, help="configure file name")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
    )
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    cparser = ConfigParser()
    cparser.read(args.config)
    shear_value = cparser.getfloat("distortion", "shear_value")
    gver = cparser.get("distortion", "g_test")
    print("Testing for %s . " % gver)
    worker = Worker(args.config, gver=gver)
    refs = list(range(args.minId, args.maxId))
    outs = []
    for r in pool.map(worker.run, refs):
        outs.append(r)

    outs = np.stack(outs)
    nsims = outs.shape[0]
    # names= [('cut','<f8'), ('de','<f8'), ('eA','<f8')
    # ('res','<f8')]
    res = np.average(outs, axis=0)
    err = np.std(outs, axis=0)
    mbias = (res[1] / res[3] / 2.0 - shear_value) / shear_value
    merr = (err[1] / res[3] / 2.0) / shear_value / np.sqrt(nsims)
    cbias = res[2] / res[3]
    cerr = err[2] / res[3] / np.sqrt(nsims)

    print("Separate galaxies into %d bins: %s" % (len(res[0]), res[0]))
    print("Multiplicative biases for those bins are: ", mbias)
    print("Errors are: ", merr)
    print("Additive biases for those bins are: ", cbias)
    print("Errors are: ", cerr)
    del worker
    pool.close()
