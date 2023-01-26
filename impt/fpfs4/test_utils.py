import numpy as np


def initialize_FPFS(fs, snlist, params):
    """Initialize FPFS catalog class
    Args:
        fs:                 fpfs.catalog.summary_stats
        snlist (ndarray):   ndarray of weight names
        params:             impt.fpfs params
    """
    cutsig = []
    cut = []
    for sn in snlist:
        if sn == "detect2":
            cutsig.append(params.sigma_v)
            cut.append(params.lower_v)
        elif sn == "M00":
            cutsig.append(params.sigma_m00)
            cut.append(params.lower_m00)
        elif sn == "R2":
            cutsig.append(params.sigma_r2)
            cut.append(params.lower_r2)
    cutsig = np.array(cutsig)
    cut = np.array(cut)
    fs.clear_outcomes()
    fs.update_selection_weight(snlist, cut, cutsig)
    fs.update_selection_bias(snlist, cut, cutsig)
    fs.update_ellsum()
    return fs
