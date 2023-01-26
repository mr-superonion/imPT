# lenspt autodiff pipeline
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
"""This unit test checks whether lenspt can initialize lenspt catalog
successfully
"""
import os
import fitsio

import impt
from impt.fpfs.default import col_names

test_fname = os.path.join(
    impt.fpfs.__data_dir__,
    "fpfs-cut32-0000-g1-0000.fits",
)


def test_catalog():
    print("testing for catalog initialization")
    data = fitsio.read(test_fname)[col_names]
    ndata = len(data)
    ncol = len(data.dtype.names)
    cat = impt.fpfs.read_catalog(test_fname)
    assert cat.shape == (ndata, ncol), "Catalog has incorrect shape"
    return


if __name__ == "__main__":
    test_catalog()
