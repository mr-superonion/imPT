# impt autodiff pipline
# Copyright 20221222 Xiangchong Li.
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
# python lib

# This file tells the default structure of the data
indexes = {
    "m00": 0,
    "m20": 1,
    "m22c": 2,
    "m22s": 3,
    "m40": 4,
    "m42c": 5,
    "m42s": 6,
    "m60": 7,
    "v0": 8,
    "v1": 9,
    "v2": 10,
    "v3": 11,
    "v4": 12,
    "v5": 13,
    "v6": 14,
    "v7": 15,
    "v0_g1": 16,
    "v1_g1": 17,
    "v2_g1": 18,
    "v3_g1": 19,
    "v4_g1": 20,
    "v5_g1": 21,
    "v6_g1": 22,
    "v7_g1": 23,
    "v0_g2": 24,
    "v1_g2": 25,
    "v2_g2": 26,
    "v3_g2": 27,
    "v4_g2": 28,
    "v5_g2": 29,
    "v6_g2": 30,
    "v7_g2": 31,
}


col_names = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
    "fpfs_M60",
    "fpfs_v0",
    "fpfs_v1",
    "fpfs_v2",
    "fpfs_v3",
    "fpfs_v4",
    "fpfs_v5",
    "fpfs_v6",
    "fpfs_v7",
    "fpfs_v0r1",
    "fpfs_v1r1",
    "fpfs_v2r1",
    "fpfs_v3r1",
    "fpfs_v4r1",
    "fpfs_v5r1",
    "fpfs_v6r1",
    "fpfs_v7r1",
    "fpfs_v0r2",
    "fpfs_v1r2",
    "fpfs_v2r2",
    "fpfs_v3r2",
    "fpfs_v4r2",
    "fpfs_v5r2",
    "fpfs_v6r2",
    "fpfs_v7r2",
]

ncol = len(indexes)
npeak = 8
