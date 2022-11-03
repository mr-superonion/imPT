import numpy as np
from lensPT import observable

x_test=np.array(
        [(1,2,3),(3,4,5), (5,6,7)],
        [('fpfs_M00', '<f8'),
        ('fpfs_M22c', '<f8'),
        ('fpfs_M22s', '<f8')],
        )

e1_lpt = observable.fpfs_e1_Li2018(Const=1)
