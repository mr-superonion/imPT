import os
from setuptools import setup, find_packages

# version of the package
__version__ = ""
fname = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "lensPT", "__version__.py"
)
with open(fname, "r") as ff:
    exec(ff.read())


# scripts = [
#     'bin/a.py',
# ]

setup(
    name="lensPT",
    version=__version__,
    description="Auto-diff Estimator of Lensing Perturbations",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "schwimmbad",
        "jax",
        "fpfs",
    ],
    packages=find_packages(),
    # scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/mr-superonion/lensPT/",
)
