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
import numpy as np
import astropy.io.fits as pyfits

from argparse import ArgumentParser
from configparser import ConfigParser


parser = ArgumentParser(description="fpfs procsim")
parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="configure file name",
)

args = parser.parse_args()
cparser = ConfigParser()
cparser.read(args.config)
sum_dir = cparser.get("procsim", "sum_dir")
shear = cparser.getfloat("distortion", "shear_value")

for mag in [24, 24.5, 25, 25.5, 26.0, 26.5]:
    fname = "%s/bin_%.1f.fits" % (sum_dir, mag)
    if not os.path.isfile(fname):
        continue
    print("magnitude is: %.1f" % mag)
    a = pyfits.getdata(fname)
    a = a[np.argsort(a[:, 0])]
    nsim = a.shape[0]
    msk = np.isnan(a[:, 3])
    b = np.average(a, axis=0)
    c = np.std(a, axis=0)
    print(b[1] / b[3] / shear / 2.0 - 1, c[1] / b[3] / shear / 2.0 / np.sqrt(nsim))
    print(b[2] / b[3], c[2] / b[3] / np.sqrt(nsim))
