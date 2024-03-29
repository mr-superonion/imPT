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
import glob
import numpy as np
import astropy.io.fits as pyfits

from argparse import ArgumentParser
from configparser import ConfigParser


parser = ArgumentParser(description="fpfs summary")
parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="configure file name",
)

args = parser.parse_args()
cparser = ConfigParser()
cparser.read(args.config)
sum_dir = cparser.get("files", "sum_dir")
shear = cparser.getfloat("distortion", "shear_value")

flist = glob.glob("%s/bin_*.*.fits" % (sum_dir))
for fname in flist:
    mag = fname.split("/")[-1].split("bin_")[-1].split(".fits")[0]
    print("magnitude is: %s" % mag)
    a = pyfits.getdata(fname)
    a = a[np.argsort(a[:, 0])]
    nsim = a.shape[0]
    msk = np.isnan(a[:, 3])
    b = np.average(a, axis=0)
    c = np.std(a, axis=0)
    p = 99.7
    p2 = (100 - p) / 2.0
    p3 = p + p2
    print(b[1] / b[3] / shear / 2.0 - 1)
    print(
        (np.percentile(a[:, 1], p2) - np.average(a[:, 1]))
        / np.average(a[:, 3])
        / shear
        / 2.0
        / np.sqrt(nsim),
        (np.percentile(a[:, 1], p3) - np.average(a[:, 1]))
        / np.average(a[:, 3])
        / shear
        / 2.0
        / np.sqrt(nsim),
    )
    print(b[2] / b[3])
    print(
        (np.percentile(a[:, 2], p2) - np.average(a[:, 2]))
        / np.average(a[:, 3])
        / np.sqrt(nsim),
        (np.percentile(a[:, 1], p3) - np.average(a[:, 1]))
        / np.average(a[:, 3])
        / np.sqrt(nsim),
    )
