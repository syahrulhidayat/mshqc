# MSHQC - Multi-State High-Quality Calculations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mshqc.svg)](https://badge.fury.io/py/mshqc)

**MSHQC** is a high-performance, open-source quantum chemistry library implementing modern electronic structure methods with seamless Python bindings. Designed for advanced research in physics and materials science, it focuses on both accuracy and extreme computational efficiency. The C++ core is heavily optimized, leveraging hardware-specific vectorization (AVX2/FMA) and Interprocedural Optimization (IPO/LTO) to maximize floating-point throughput.

---

## 🌟 Features

### 1. Self-Consistent Field (SCF)
- Restricted, Unrestricted, and Restricted Open-shell Hartree-Fock (RHF, UHF, ROHF)
- DIIS convergence acceleration
- Cholesky Decomposition Variants (CD-RHF, CD-UHF, CD-ROHF)

### 2. Perturbation Theory (MPn)
- Møller-Plesset MP2 & MP3 (Restricted and Unrestricted)
- Orbital-Optimized MP2/MP3 (OMP2/OMP3)
- Cholesky Decomposition Variants (CD-RMP2/3, CD-UMP2/3, CD-OMP2/3)

### 3. Multi-Configurational Methods (MCSCF)
- Complete Active Space SCF (CASSCF) & State-Averaged CASSCF (SA-CASSCF)
- Complete Active Space Perturbation Theory 2nd order (CASPT2)
- Cholesky Decomposition Variants (CD-CASSCF, CD-SA-CASSCF)
- Cholesky Decomposition Perturbation Theories (CD-CASPT2, CD-SA-CASPT2, CD-SA-CASPT3)

### 4. Advanced Tooling
- **Integral Transformations**: Cholesky decomposition for ERI and Four-Index Transformations
- **Gradients**: Analytical and numerical gradients, and geometry optimization
- **Properties**: Natural orbitals, transition density matrices, and one-particle density matrices (OPDM)

---

## 🚀 The "Ultimate" Computational Engine & Dependencies

Under the hood, MSHQC bridges C++17 and Python using a deeply integrated scientific computing stack. The build system (`CMakeLists.txt`) strictly requires the following dependencies to handle massive out-of-core calculations and tensor operations:

- **Compiler**: GCC 7+ or Clang 5+ (must support C++17, AVX2, FMA, and IPO/LTO)
- **CMake**: Version 3.14 or higher
- **Python Binding**: `nanobind` (lightweight, highly efficient C++/Python interface)
- **Multi-threading**: `OpenMP` (native shared-memory parallelization)
- **Integral Engine**: `libcint` (high-performance analytical electron repulsion integrals)
- **Tensor Contraction**: `TBLIS` (fast tensor contraction avoiding explicit multidimensional array transposition)
- **Out-of-Core Data**: `HDF5` (C and C++ bindings required for massive tensor storage and ABI stability)
- **Linear Algebra (Templates)**: `Eigen3` (modern heavily vectorized matrix operations)
- **BLAS Backend**: `BLIS` (preferred) or `OpenBLAS`
- **LAPACK Backend**: `libflame` (preferred) or standard `LAPACK` + `LAPACKE` (C interface)

---

## 📦 Installation

### Option A: Install Pre-compiled Wheels via PyPI (Recommended)

MSHQC is continuously tested and deployed via GitHub Actions. Pre-compiled, `manylinux`-compatible wheels are available for Python 3.10 through 3.13.

```bash
pip install mshqc
```

### Option B: Compiling from Source (Super-Turbo Mode)

For maximum performance, compile MSHQC locally to leverage `-march=native` optimizations for your specific CPU architecture. An isolated Conda environment is highly recommended to manage the complex C++ libraries.

**1. Conda Environment Setup**

Install the complete dependency stack via `conda-forge`:

```bash
# Create and activate environment
conda create -n mshqc_env python=3.12
conda activate mshqc_env

# Install critical C++ libraries and build tools
conda install -y -c conda-forge \
    cmake make compilers eigen pkg-config \
    "hdf5=1.14.3" pip libcint tblis liblapacke openblas

# Install Python-level build requirements
python -m pip install build wheel nanobind numpy scipy
```

**2. Build and Install**

Clone the repository and install it in editable mode. The CMake build system will automatically detect the Conda environment and link against the optimized C++ libraries.

```bash
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Compile and install the Python bindings
pip install -e .
```

---

## 💻 Quick Start (Anti-Crash Configuration)

> ⚠️ **CRITICAL WARNING**: MSHQC and scientific libraries like `numpy` both utilize multithreaded BLAS backends (e.g., OpenBLAS). To prevent catastrophic CPU thread collisions or segmentation faults, you must configure thread limits at the OS level **before** importing any libraries.

```python
import os

# 1. Lock thread counts BEFORE importing to prevent OpenBLAS collisions
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TBLIS_ARCH"] = "x86_64"

# 2. Import MSHQC FIRST
import mshqc

# 3. Import other scientific libraries
import numpy as np

# Set up your calculation parameters
print(f"MSHQC Version: {mshqc.__version__}")
```

---

## 🛠️ Development & CI/CD

This project utilizes GitHub Actions for continuous integration. Upon pushing to specified branches or tagging releases, the pipeline automatically:

1. Compiles the C++ core with Universal HPC optimizations (`-O3 -mavx2 -mfma`)
2. Generates Python bindings via `nanobind`
3. Repairs Linux binaries using `auditwheel` to ensure cross-platform ABI compatibility
4. Deploys the wheels directly to PyPI and the public-facing repository

---

## 🐛 Bug Reports & Contributions

If you encounter any issues, compilation errors, or have feature requests, please report them on the Issue Tracker. Code contributions, bug fixes, and documentation improvements are highly appreciated.

---

## 📄 License & Author

- **Author**: Muhamad Syahrul Hidayat
- **License**: This project is licensed under the MIT License. See the `LICENSE` file for details.
