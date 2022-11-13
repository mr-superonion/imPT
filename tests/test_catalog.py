# lensPT autodiff pipeline
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
import fitsio
import lensPT as lpt


def test_catalog():
    print("testing for catalog initialization")
    data = fitsio.read("./fpfs-cut32-0000-g1-0000.fits")
    colnames = [
        "fpfs_M00",
        "fpfs_M20",
        "fpfs_M22c",
        "fpfs_M22s",
        "fpfs_M40",
        "fpfs_M42c",
        "fpfs_M42s",
    ]
    lpt.observable.Catalog(data, mode_names=colnames)
    lpt.observable.Catalog(data)
    lpt.observable.Catalog("./fpfs-cut32-0000-g1-0000.fits", mode_names=colnames)
    cat = lpt.observable.Catalog("./fpfs-cut32-0000-g1-0000.fits")
    cat.mode_names
    cat.data
    return


if __name__ == "__main__":
    test_catalog()