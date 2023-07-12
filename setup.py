import os
from setuptools import setup, find_packages

# version of the package
__version__ = ""
fname = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "impt", "__version__.py"
)
with open(fname, "r") as ff:
    exec(ff.read())


scripts = [
    "bin/impt_config",
    "bin/impt_summary_hsc.py",
    "bin/impt_summary_desc_future.py",
    "bin/impt_std_future.py",
    "bin/impt_std_future_snr.py",
    "bin/impt_std_future_from_image.py",
    "bin/impt_make_plots.py",
]

setup(
    name="impt",
    version=__version__,
    description="Auto-diff Estimator of Lensing Perturbations",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "schwimmbad",
        "jax>=0.4.9",
        "jaxlib>=0.4.9",
        "flax",
        "fitsio",
        "pre-commit",
    ],
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/mr-superonion/imPT/",
)
