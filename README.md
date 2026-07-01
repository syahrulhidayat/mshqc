# MSHQC - Multi-State High-Quality Calculations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.8-3.13](https://img.shields.io/badge/Python-3.8--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mshqc.svg)](https://badge.fury.io/py/mshqc)

**MSHQC** is a high-performance, open-source quantum chemistry library implementing modern electronic structure methods with seamless Python bindings. Designed for both accuracy and extreme computational efficiency, it leverages a highly optimized C++ core utilizing hardware-specific vectorization (AVX2/FMA).

## 🚀 The "Ultimate" Computational Engine
Under the hood, MSHQC is built on a heavily optimized scientific computing stack to handle extremely large-scale out-of-core calculations:
* **libcint**: High-performance analytical integral engine for electron repulsion integrals (ERI).
* **TBLIS**: Fast tensor contraction framework that avoids explicit multidimensional array transposition.
* **HDF5**: Out-of-core data handling for massive tensor storage (specifically optimized for ABI stability).
* **Eigen3**: Modern C++ template library for linear algebra, heavily vectorized.
* **BLIS / OpenBLAS & LAPACKE**: Multithreaded standard linear algebra backend.
* **nanobind**: Lightweight, highly efficient Python/C++ binding interface (replacing Pybind11).
* **Jemalloc** *(Optional)*: Scalable and fragmentation-resistant memory allocator.

## 🌟 Features

### 1. Self-Consistent Field (SCF)
* Restricted, Unrestricted, and Restricted Open-shell Hartree-Fock (RHF, UHF, ROHF).
* DIIS convergence acceleration.
* Cholesky Decomposition Variants (CD-RHF, CD-UHF, CD-ROHF).

### 2. Perturbation Theory (MPn)
* Møller-Plesset MP2 & MP3 (Restricted and Unrestricted).
* Orbital-Optimized MP2/MP3 (OMP2/OMP3).
* Cholesky Decomposition Variants (CD-RMP2/3, CD-UMP2/3, CD-OMP2/3).

### 3. Multi-Configurational Methods (MCSCF)
* Complete Active Space SCF (CASSCF) & State-Averaged CASSCF (SA-CASSCF).
* Complete Active Space Perturbation Theory 2nd order (CASPT2).
* Cholesky Decomposition Variants (CD-CASSCF, CD-SA-CASSCF).
* Cholesky Decomposition Perturbation Theories (CD-CASPT2, CD-SA-CASPT2, CD-SA-CASPT3).

### 4. Advanced Tooling
* **Integral Transformations**: Cholesky decomposition for ERI and Four-Index Transformations.
* **Gradients**: Analytical and numerical gradients, and geometry optimization.
* **Properties**: Natural orbitals, transition density matrices, and one-particle density matrices (OPDM).

---

## 📦 Installation

### Option A: Install Pre-compiled Wheels via PyPI (Recommended)
MSHQC is continuously tested and deployed via GitHub Actions. Pre-compiled, `manylinux`-compatible wheels are available for Python 3.10 through 3.13.

```bash
pip install mshqc
Option B: Compiling from Source (Super-Turbo Mode)

If you want to compile MSHQC locally to leverage -march=native optimizations for your specific CPU architecture, we highly recommend using an isolated Conda environment to manage the complex C++ library dependencies.
1. System Requirements

    C++ Compiler: GCC 7+ or Clang 5+ (Must support C++17, AVX2, and FMA).

    CMake: Version 3.18 or higher.

    Build Tools: Ninja, Make, ccache (recommended).

2. Conda Environment Setup

To prevent linking errors, install the complete "Ultimate Stack" via conda-forge:
Bash

# Create and activate environment
conda create -n mshqc_env python=3.12
conda activate mshqc_env

# Install critical C++ libraries and build tools
conda install -y -c conda-forge \
    cmake make compilers eigen pkg-config \
    "hdf5=1.14.3" pip libcint tblis liblapacke openblas

# Install Python-level build requirements
python -m pip install build wheel nanobind numpy scipy

3. Build and Install

Clone the repository and install it in editable mode:
Bash

git clone [https://github.com/syahrulhidayat/mshqc.git](https://github.com/syahrulhidayat/mshqc.git)
cd mshqc

# Compile and install the Python bindings
pip install -e .

(Note: The build system automatically detects the Conda environment and links against the optimized versions of libcint, TBLIS, and HDF5).
💻 Quick Start
Python

import mshqc
import numpy as np

# Set up your calculation parameters
# (Refer to the official documentation for detailed API usage)
print(f"MSHQC Version: {mshqc.__version__}")

🛠️ Development & CI/CD

This project utilizes GitHub Actions for continuous integration. Upon pushing to specified branches or tagging releases, the pipeline automatically:

    Compiles the C++ core with Universal HPC optimizations.

    Generates Python bindings via nanobind.

    Repairs Linux binaries using auditwheel to ensure cross-platform compatibility.

    Deploys the wheels to PyPI and a public facing repository.

🐛 Bug Reports & Contributions

If you encounter any issues, compile errors, or have feature requests, please report them on the Issue Tracker. Code contributions, bug fixes, and documentation improvements are highly appreciated.
📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
