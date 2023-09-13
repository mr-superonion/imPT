# imPT
----
[![tests](https://github.com/mr-superonion/lensPT/actions/workflows/tests.yml/badge.svg)](https://github.com/mr-superonion/lensPT/actions/workflows/tests.yml)
[![pypi](https://github.com/mr-superonion/imPT/actions/workflows/pypi.yml/badge.svg)](https://github.com/mr-superonion/imPT/actions/workflows/pypi.yml)
[![Documentation Status](https://readthedocs.org/projects/impt/badge/?version=latest)](https://impt.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Fast estimator for Lensing Perturbation (`impt`) on astronomical images using
the auto differentiating function of jax. The code need to be combined with
estimator (e.g. [FPFS](https://github.com/mr-superonion/FPFS)) compressing image
to catalog. It is a simple code to infer shear from the catalog with correction for noise bias

----

## Installation

For stable (old) version, which have not been updated:
```shell
pip install impt
```

For developers, please clone the repository:
```shell
git clone https://github.com/mr-superonion/imPT.git
pip install -e . --user
```

## Summary
For the first version of `imPT`, we implement `FPFS` and use `imPT` to auto
differentiate `FPFS`.
