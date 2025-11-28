import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
from pathlib import Path

try:
    import pybind11
except ImportError:
    # pybind11 will be installed by pip if not available
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.12"])
    import pybind11

# Define Python extension class
ext_modules = [
    Extension(
        "mshqc._core",
        sources=["python/bindings.cc"],
        include_dirs=[
            "include",
            "src",
            numpy.get_include(),
            pybind11.get_include(),
        ],
        language="c++",
    ),
]

# Setup configuration
setup(
    name="mshqc",
    version="0.1.0",
    description="Python bindings for MSH-QC quantum mechanics library",
    long_description=Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Muhamad Sahrul Hidayat",
    license="MIT",
    url="https://github.com/syahrulhidayat/mshqc",
    packages=["mshqc"],
    package_dir={"": "python"},
    package_data={"mshqc": ["py.typed"]},
    include_package_data=True,
    ext_modules=ext_modules,
    python_requires=">=3.8",
    install_requires=["numpy>=1.22", "pybind11>=2.12"],
    zip_safe=False,
)