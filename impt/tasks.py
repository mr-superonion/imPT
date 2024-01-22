import os
import gc
import jax
import time
import impt
import fitsio
import logging
import numpy as np
from impt.fpfs.future import prepare_func_e
from configparser import ConfigParser, ExtendedInterpolation

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


class MeasureShearSimulationTask(object):
    def __init__(
        self,
        config_name,
        magcut=27.0,
        min_id=0,
        max_id=1000,
        ncores=1,
    ):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        # survey parameter
        nids = max_id - min_id
        self.n_per_c = nids // ncores
        self.mid = nids % ncores
        self.min_id = min_id
        self.max_id = max_id
        self.rest_list = list(np.arange(ncores * self.n_per_c, nids) + min_id)
        logging.info("number of files per core is: %d" % self.n_per_c)

        # setup processor
        self.catdir = cparser.get("files", "cat_dir")
        self.sum_dir = cparser.get("files", "sum_dir")
        os.makedirs(self.sum_dir, exist_ok=True)
        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
        self.cov_mat = fitsio.read(self.ncov_fname)
        # FPFS parameters
        self.ratio = cparser.getfloat("FPFS", "ratio")
        self.c0 = cparser.getfloat("FPFS", "c0")
        self.c2 = cparser.getfloat("FPFS", "c2")
        self.alpha = cparser.getfloat("FPFS", "alpha")
        self.beta = cparser.getfloat("FPFS", "beta")
        self.noise_rev = cparser.getboolean("FPFS", "noise_rev", fallback=True)
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")
        self.band = cparser.get("survey", "band")
        # This task change the cut on one observable and see how the biases
        # changes.
        # Here is  the observable used for test
        self.upper_mag = magcut
        self.lower_m00 = 10 ** ((self.magz - self.upper_mag) / 2.5)
        # setup WL distortion parameter
        self.g_comp = cparser.getint("FPFS", "g_component_measure", fallback=1)
        assert self.g_comp in [1, 2], "The g_comp in configure file is not supported"
        self.gver = cparser.get("distortion", "g_version")
        self.ofname = os.path.join(
            self.sum_dir,
            "bin_%s.fits" % (self.upper_mag),
        )
        return

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        assert os.path.isfile(
            in_nm
        ), "Cannot find input galaxy shear catalogs : %s " % (in_nm)
        mm = impt.fpfs.read_catalog(in_nm)
        # noise bias

        def fune(carry, ss):
            y = e1._obs_func(ss) - enoise._obs_func(ss)
            return carry + y, y

        def funr(carry, ss):
            y = res1._obs_func(ss) - rnoise._obs_func(ss)
            return carry + y, y

        e1_sum, _ = jax.lax.scan(fune, 0.0, mm)
        r1_sum, _ = jax.lax.scan(funr, 0.0, mm)
        del mm
        gc.collect()
        return e1_sum, r1_sum

    def get_range(self, icore):
        ibeg = self.min_id + icore * self.n_per_c
        iend = min(ibeg + self.n_per_c, self.max_id)
        id_range = list(range(ibeg, iend))
        if icore < len(self.rest_list):
            id_range.append(self.rest_list[icore])
        return id_range

    def run(self, icore):
        start_time = time.time()
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 4))
        logging.info("start core: %d, with id: %s" % (icore, id_range))
        for icount, ifield in enumerate(id_range):
            for irot in range(2):
                e1, enoise, res1, rnoise = prepare_func_e(
                    cov_mat=self.cov_mat,
                    snr_min=self.lower_m00 / np.sqrt(self.cov_mat[0, 0]),
                    ratio=self.ratio,
                    c0=self.c0,
                    c2=self.c2,
                    alpha=self.alpha,
                    beta=self.beta,
                    noise_rev=self.noise_rev,
                    g_comp=self.g_comp,
                )
                in_nm1 = os.path.join(
                    self.catdir,
                    "src-%05d_%s-0_rot%d_%s.fits"
                    % (ifield, self.gver, irot, self.band),
                )
                e1_1, r1_1 = self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)
                in_nm2 = os.path.join(
                    self.catdir,
                    "src-%05d_%s-1_rot%d_%s.fits"
                    % (ifield, self.gver, irot, self.band),
                )
                e1_2, r1_2 = self.get_sum_e_r(in_nm2, e1, enoise, res1, rnoise)
                out[icount, 0] = ifield
                out[icount, 1] = out[icount, 1] + (e1_2 - e1_1)
                out[icount, 2] = out[icount, 2] + (e1_1 + e1_2) / 2.0
                out[icount, 3] = out[icount, 3] + (r1_1 + r1_2) / 2.0
                del e1, enoise, res1, rnoise
                # jax.clear_backends()
                # jax.clear_caches()
                gc.collect()
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 4.0
        logging.info(f"Elapsed time: {elapsed_time} seconds")
        return out
