import os
from setuptools import setup, find_packages

# version of the package
__version__ = ""
fname = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "gimpt", "__version__.py"
)
with open(fname, "r") as ff:
    exec(ff.read())


scripts = [
    "bin/gimpt_config",
]

setup(
    name="gimpt",
    version=__version__,
    description="Auto-diff Estimator of Lensing Perturbations",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "schwimmbad",
        "jax",
        "flax",
        "fpfs",
        "fitsio",
    ],
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/mr-superonion/lensPT/",
)
