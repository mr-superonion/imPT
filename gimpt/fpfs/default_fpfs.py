# gimpt autodiff pipline
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
m00 = 0
m20 = 1
m22c = 2
m22s = 3
m40 = 4
m42c = 5
m42s = 6
# peak modes
v0 = 7
v1 = 8
v2 = 9
v3 = 10
v4 = 11
v5 = 12
v6 = 13
v7 = 14
# shear response of peak modes
# g1
v0_g1 = 15
v1_g1 = 16
v2_g1 = 17
v3_g1 = 18
v4_g1 = 19
v5_g1 = 20
v6_g1 = 21
v7_g1 = 22
# g2
v0_g2 = 23
v1_g2 = 24
v2_g2 = 25
v3_g2 = 26
v4_g2 = 27
v5_g2 = 28
v6_g2 = 29
v7_g2 = 30

ncol = 31
ind_trunc = -16

col_names = [
    "m00",
    "m20",
    "m22c",
    "m22s",
    "m40",
    "m42c",
    "m42s",
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v0_g1",
    "v1_g1",
    "v2_g1",
    "v3_g1",
    "v4_g1",
    "v5_g1",
    "v6_g1",
    "v7_g1",
    "v0_g2",
    "v1_g2",
    "v2_g2",
    "v3_g2",
    "v4_g2",
    "v5_g2",
    "v6_g2",
    "v7_g2",
]
