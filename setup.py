import os
import sys
import subprocess
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import numpy

# Version
__version__ = "0.1.0"

# Determine if we can find libint2
def find_libint2():
    """Find libint2 library and include paths"""
    libint2_found = False
    libint2_includes = []
    libint2_libs = []
    
    # Try common installation paths
    include_paths = [
        "/usr/include/libint2",
        "/usr/local/include/libint2",
        "/usr/include", 
        "/usr/local/include"
    ]
    
    library_paths = [
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib",
        "/usr/local/lib64"
    ]
    
    # Find include directory
    for inc_path in include_paths:
        if os.path.exists(os.path.join(inc_path, "libint2.hpp")):
            libint2_includes.append(inc_path)
            break
    else:
        # Find libint2.hpp in any subdirectory
        for inc_path in include_paths:
            for root, _, files in os.walk(inc_path):
                if "libint2.hpp" in files:
                    libint2_includes.append(root)
                    break
            if libint2_includes:
                break
    
    # Find library
    for lib_path in library_paths:
        if os.path.exists(os.path.join(lib_path, "libint2.so")) or \
           os.path.exists(os.path.join(lib_path, "libint2.dylib")) or \
           os.path.exists(os.path.join(lib_path, "libint2.a")):
            libint2_libs.append(lib_path)
            break
    
    if libint2_includes and libint2_libs:
        libint2_found = True
    
    return libint2_found, libint2_includes, libint2_libs

# Determine if we can find libcint
def find_libcint():
    """Find libcint library and include paths"""
    libcint_found = False
    libcint_includes = []
    libcint_libs = []
    
    # Try common installation paths
    include_paths = [
        "/usr/include/libcint",
        "/usr/local/include/libcint",
        "/usr/include", 
        "/usr/local/include"
    ]
    
    library_paths = [
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib",
        "/usr/local/lib64"
    ]
    
    # Find include directory
    for inc_path in include_paths:
        if os.path.exists(os.path.join(inc_path, "cint.h")):
            libcint_includes.append(inc_path)
            break
    else:
        # Find cint.h in any subdirectory
        for inc_path in include_paths:
            for root, _, files in os.walk(inc_path):
                if "cint.h" in files:
                    libcint_includes.append(root)
                    break
            if libcint_includes:
                break
    
    # Find library
    for lib_path in library_paths:
        if os.path.exists(os.path.join(lib_path, "libcint.so")) or \
           os.path.exists(os.path.join(lib_path, "libcint.dylib")) or \
           os.path.exists(os.path.join(lib_path, "libcint.a")) or \
           os.path.exists(os.path.join(lib_path, "libcint.dll")):
            libcint_libs.append(lib_path)
            break
    
    if libcint_includes and libcint_libs:
        libcint_found = True
    
    return libcint_found, libcint_includes, libcint_libs

# Find required packages
print("Finding dependencies...")
libint2_found, libint2_includes, libint2_libs = find_libint2()
libcint_found, libcint_includes, libcint_libs = find_libcint()

if not libint2_found:
    print("Warning: libint2 not found, MSH-QC will not be fully functional")
    print("Please install libint2 before proceeding")
    
print(f"Libint2 found: {libint2_found}")
print(f"Libcint found: {libcint_found}")

# Include directories
include_dirs = [
    "include",
    "python",
    "src",
    numpy.get_include(),
] + libint2_includes + libcint_includes

# Library directories
library_dirs = libint2_libs + libcint_libs

# Compile arguments
compile_args = ["-std=c++17", "-O3", "-Wall", "-shared", "-fPIC"]

# Link arguments
link_args = []

# Determine libraries to link
libraries = ["mshqc", "int2"]
if libcint_found:
    libraries.append("cint")

# Add OpenMP if available
try:
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")
except:
    print("OpenMP not found or not supported")

# Source files for the extension module
ext_sources = [os.path.join("python", "bindings.cc")]

# Find all MSH-QC source files
source_dirs = [
    "src/core",
    "src/mp2", 
    "src/mp3",
    "src/mp",
    "src/mp",
    "src/scf",
    "src/ci",
    "src/mcscf",
    "src/integrals",
    "src/integration",
    "src/validation",
    "src/gradient"
]

for src_dir in source_dirs:
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".cc"):
                ext_sources.append(os.path.join(root, file))

# Create extension module
ext_modules = [
    Pybind11Extension(
        "mshqc._core",
        ext_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    ),
]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mshqc",
    version=__version__,
    author="Muhamad Sahrul Hidayat",
    author_email="syahrul@example.com",
    description="Python bindings for MSH-QC quantum mechanics library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/syahrulhidayat/mshqc",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["mshqc"],
    package_dir={"": "python"},
    package_data={"mshqc": ["*.pyi", "data/basis/*.gbs"]},
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "sphinx>=3.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    zip_safe=False,
)