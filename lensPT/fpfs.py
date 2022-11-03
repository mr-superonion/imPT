# lensPT autodiff pipline
# Copyright 20221031 Xiangchong Li.
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

# This is only a simple example
# You can define your own observable function


from lensPT.observable import Observable

class E1(Observable):
    def __init__(self, Const):
        super(E1, self).__init__(Const=Const)
        self.mode_names = [
                "fpfs_M22c",
                "fpfs_M00",
                ]
        return

    def base_func(self, x):
        e1 = x[self.aind("fpfs_M22c")] \
            / ( x[self.aind("fpfs_M00")] + self.meta["Const"] )
        return e1


class E2(Observable):
    def __init__(self, Const):
        super(E2, self).__init__(Const=Const)
        self.mode_names = [
                "fpfs_M22s",
                "fpfs_M00",
                ]
        return

    def base_func(self, x):
        e1 = x[self.aind("fpfs_M22s")] \
            / ( x[self.aind("fpfs_M00")] + self.meta["Const"])
        return e1


class Weight(Observable):

    def __init__(self, **kwargs):
        super(Weight, self).__init__()
        self.mode_names = [
                "fpfs_M22s",
                "fpfs_M00",
                ]
        return

    def base_func(self, x):
        pass
