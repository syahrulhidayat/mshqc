# MSHQC - Multi-State High-Quality Calculations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**MSHQC** is a high-performance quantum chemistry library implementing modern electronic structure methods with seamless Python bindings. Designed for both accuracy and extreme efficiency, it leverages a highly optimized C++ core utilizing hardware-specific vectorization (AVX2/FMA).

## 🌟 Features

### Electronic Structure Methods
- **Self-Consistent Field (SCF)**
  - Restricted Hartree-Fock (RHF)
  - Unrestricted Hartree-Fock (UHF)
  - Restricted Open-shell Hartree-Fock (ROHF)
  - DIIS convergence acceleration
  - Cholesky Decomposition Variants (CD-RHF, CD-UHF, CD-ROHF)

- **Møller-Plesset Perturbation Theory**
  - MP2, MP3 (Restricted and Unrestricted)
  - Orbital-Optimized MP2/MP3 (OMP2/OMP3)
  - Cholesky Decomposition Variants (CD-RMP2/3, CD-UMP2/3, CD-OMP2/3)

- **Multi-Configurational Self-Consistent Field (MCSCF)**
  - Complete Active Space SCF (CASSCF) & State-Averaged CASSCF (SA-CASSCF)
  - Complete Active Space Perturbation Theory 2nd order (CASPT2)
  - Cholesky Decomposition Variants (CD-CASSCF, CD-SA-CASSCF)
  - Cholesky Decomposition Perturbation Theories (CD-CASPT2, CD-SA-CASPT2, CD-SA-CASPT3)
    
- **Configuration Interaction (CI)** *(in progress)*
  - Configuration Interaction Singles (CIS)
  - Configuration Interaction Singles and Doubles (CISD)
  - Configuration Interaction Singles, Doubles, and Triples (CISDT)
  - Full Configuration Interaction (FCI)
  - Multireference CI (MRCI)
  - CIPSI (Configuration Interaction by Perturbation with Selection Iteratively)

### Advanced Capabilities
- **Integral Transformations**: Cholesky decomposition for electron repulsion integrals (ERI) and Four-Index Transformation.
- **Gradient and Optimization**: Analytical/numerical gradients and geometry optimization.
- **Analysis Tools**: Natural orbitals, transition density matrices, one-particle density matrices (OPDM), and wavefunction analysis.

## 🚀 The Computational Engine
Under the hood, MSHQC is built on an ultimate scientific computing stack to handle extremely large-scale calculations:
- **libcint**: High-performance analytical integral engine.
- **TBLIS**: Fast tensor contraction without explicit transposition.
- **HDF5**: Out-of-core data handling for massive tensor storage.
- **BLIS/LAPACKE**: Multithreaded linear algebra backend.
- **nanobind**: Lightweight, highly efficient Python/C++ binding interface.

---

## 📋 Installation

### 1. Install via PyPI (Recommended)
Pre-compiled wheels are available for standard usage without requiring a local C++ compilation environment.
```bash
pip install mshqc
2. Building from Source (Using Conda)

For development or enabling architecture-specific super-turbo optimizations (-march=native), compiling from source is recommended.

Prerequisites:

    C++ Compiler: GCC 7+ or Clang 5+ (C++17 support required)

    CMake: 3.18 or higher

    Conda/Miniconda

Steps:
Bash

# 1. Create and activate a conda environment
conda create -n mshqc_env python=3.12
conda activate mshqc_env

# 2. Install build dependencies (Sync with psi4env standards)
conda install -c conda-forge \
    cmake make compilers eigen pkg-config \
    "hdf5=1.14.3" pip libcint tblis liblapacke openblas

# 3. Clone the repository
git clone [https://github.com/syahrulhidayat/mshqc.git](https://github.com/syahrulhidayat/mshqc.git)
cd mshqc

# 4. Build and install the package
pip install -e .

💻 Quick Start
Python

import mshqc
import numpy as np

# Example initialization (pseudo-code)
# Setup molecule and compute ground state energy

🐛 Bug Reports & Contributions

If you encounter any issues or have feature requests, please check the Issue Tracker. Contributions are welcome!
📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
