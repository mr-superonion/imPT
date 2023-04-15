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
import impt
import time
import fitsio
import schwimmbad
import numpy as np
import jax.numpy as jnp

from argparse import ArgumentParser
from configparser import ConfigParser

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ[
    "XLA_FLAGS"
] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"


class Worker(object):
    def __init__(self, config_name, gver="g1"):
        cparser = ConfigParser()
        cparser.read(config_name)
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")
        cov_fname = cparser.get("FPFS", "mode_cov_name")
        self.cov_mat = jnp.array(fitsio.read(cov_fname))

        # setup processor
        self.indir = cparser.get("IO", "input_dir")
        self.outdir = cparser.get("IO", "output_dir")
        os.makedirs(self.outdir, exist_ok=True)

        self.rcut = cparser.getint("FPFS", "rcut")
        # This task change the cut on one observable and see how the biases changes.
        # Here is  the observable used for test
        self.upper_mag = cparser.getfloat("FPFS", "cut_mag")
        self.lower_m00 = 10 ** ((self.magz - self.upper_mag) / 2.5)
        self.lower_r2 = cparser.getfloat("FPFS", "cut_r2")

        if not os.path.exists(self.indir):
            raise FileNotFoundError("Cannot find input directory: %s!" % self.indir)
        print("The input directory for galaxy shear catalogs is %s. " % self.indir)
        # setup WL distortion parameter
        self.gver = gver
        return

    def prepare_functions(self):
        params = impt.fpfs.FpfsParams(
            Const=20,
            lower_m00=self.lower_m00,
            sigma_m00=0.2,
            lower_r2=self.lower_r2,
            upper_r2=200,
            sigma_r2=0.4,
            sigma_v=0.2,
        )
        funcnm = "ss2"
        e1_impt = impt.fpfs.FpfsE1(params, func_name=funcnm)
        w_det = impt.fpfs.FpfsWeightDetect(params, func_name=funcnm)
        w_sel = impt.fpfs.FpfsWeightSelect(params, func_name=funcnm)

        # ellipticity
        e1 = e1_impt * w_sel * w_det
        enoise = impt.BiasNoise(e1, self.cov_mat)
        res1 = impt.RespG1(e1)
        rnoise = impt.BiasNoise(res1, self.cov_mat)
        gc.collect()
        return e1, enoise, res1, rnoise

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        assert os.path.isfile(
            in_nm
        ), "Cannot find input galaxy shear catalogs : %s " % (in_nm)
        mm = impt.fpfs.read_catalog(in_nm)
        print("number of galaxies: %d" % len(mm))
        e1_sum = jnp.sum(e1.evaluate(mm))
        e1_sum = e1_sum - jnp.sum(enoise.evaluate(mm))

        # shear response
        r1_sum = jnp.sum(res1.evaluate(mm))
        r1_sum = r1_sum - jnp.sum(rnoise.evaluate(mm))
        del mm
        return e1_sum, r1_sum

    def run(self, ind0):
        out_nm = os.path.join(self.outdir, "%04d.fits" % ind0)
        if os.path.isfile(out_nm):
            print("Already has the output file")
            return

        start_time = time.time()
        e1, enoise, res1, rnoise = self.prepare_functions()
        pp = "cut%d" % self.rcut
        in_nm1 = os.path.join(
            self.indir, "fpfs-%s-%04d-%s-0000.fits" % (pp, ind0, self.gver)
        )
        sum_e1_1, sum_r1_1 = self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)
        gc.collect()

        in_nm2 = os.path.join(
            self.indir, "fpfs-%s-%04d-%s-2222.fits" % (pp, ind0, self.gver)
        )
        sum_e1_2, sum_r1_2 = self.get_sum_e_r(in_nm2, e1, enoise, res1, rnoise)
        del e1, enoise, res1, rnoise
        gc.collect()
        print("--- computational time: %.2f seconds ---" % (time.time() - start_time))

        out = np.zeros((4, 1))
        # names= [('cut','<f8'), ('de','<f8'), ('eA','<f8') ('res','<f8')]
        out[0, 0] = self.upper_mag
        out[1, 0] = sum_e1_2 - sum_e1_1
        out[2, 0] = (sum_e1_1 + sum_e1_2) / 2.0
        out[3, 0] = (sum_r1_1 + sum_r1_2) / 2.0
        fitsio.write(out_nm, out)
        return


def main(pool):
    cparser = ConfigParser()
    cparser.read(args.config)
    gver = cparser.get("distortion", "g_test")
    print("Testing for %s . " % gver)
    worker = Worker(args.config, gver=gver)
    refs = list(range(args.minId, args.maxId))
    for _ in pool.map(worker.run, refs):
        pass
    del worker, cparser
    pool.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="impt fpfs")
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
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    args = parser.parse_args()
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool)
