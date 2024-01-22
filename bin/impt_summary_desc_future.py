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
import fitsio
import schwimmbad
import numpy as np

from argparse import ArgumentParser
from impt.tasks import MeasureShearSimulationTask
from configparser import ConfigParser


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


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs catalog to shear")
    parser.add_argument(
        "--runid",
        default=0,
        type=int,
        help="id number, e.g. 0",
    )
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=1000,
        type=int,
        help="id number, e.g. 1000",
    )
    parser.add_argument(
        "--magcut",
        default=27.0,
        type=float,
        help="id number, e.g. 0",
    )
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
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    args = parser.parse_args()
    cparser = ConfigParser()
    cparser.read(args.config)

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    ncores = get_processor_count(pool, args)
    assert isinstance(ncores, int)
    core_list = np.arange(ncores)
    worker = MeasureShearSimulationTask(
        args.config,
        magcut=args.magcut,
        min_id=args.min_id,
        max_id=args.max_id,
        ncores=ncores,
    )

    olist = pool.map(worker.run, core_list)
    pool.close()
    outcome = np.vstack(list(olist))
    fitsio.write(worker.ofname, outcome)
