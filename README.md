# lensPT
[![Python package](https://github.com/mr-superonion/lensPT/actions/workflows/python-package.yml/badge.svg)](https://github.com/mr-superonion/lensPT/actions/workflows/python-package.yml)

Fast estimator for Lensing Perturbation (`lensPT`) from astronomical images
using the auto differentiating function of jax.

A simple code to compute the response to lensing perturbations and remove the
bias due to the perturbation from image noise.


## Installation

For developers:
```shell
git clone https://github.com/mr-superonion/lensPT.git
pip install -e . --user
```
before running code, users need to setup the environment by
```shell
source lenspt_config
```
or users can put the configure command into _dot_shrc file.

## Summary
For the first version of `lensPT`, we implement `FPFS` and use `lensPT` to auto
differentiate `FPFS`.
