# MSHQC - Modern Quantum Chemistry Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

A high-performance quantum chemistry library implementing modern electronic structure methods with Python bindings.

## 🌟 Features

### Electronic Structure Methods
- **Self-Consistent Field (SCF)**
  - Restricted Hartree-Fock (RHF)
  - Unrestricted Hartree-Fock (UHF)
  - Restricted Open-shell Hartree-Fock (ROHF)
  - DIIS convergence acceleration
  - Cholesky Unrestricted Hartree-Fock (CD-UHF)
  - Cholesky Restricted Open-shell Hartree-Fock (CD-ROHF)
  - Cholesky Restricted Hartree-Fock (CD-RHF)

- **Møller-Plesset Perturbation Theory**
  - MP2, MP3 (Restricted and Unrestricted)
  - Orbital-Optimized MP2/MP3 (OMP2/OMP3)
  - Cholesky Unrestricted MP2/MP3 (CD-UMP2/CD-UMP3)
  - Cholesky Orbital-Optimized MP2/MP3 (CD-OMP2/CD-OMP3)
  - Cholesky Restricted MP2/MP3 (CD-RMP2/CD-CD-RMP3)
    
    

- **Multi-Configurational Self-Consistent Field (MCSCF)**
  - Complete Active Space SCF (CASSCF)
  - State-Averaged CASSCF (SA-CASSCF)
  - Complete Active Space Perturbation Theory 2nd order (CASPT2)
  - Cholesky Decomposition Complete Active Space SCF (CD-CASSCF)
  - Cholesky Decomposition Complete Active Space Perturbation Theory 2nd order (CD-CASPT2)
  - Cholesky Decomposition-State-Averaged-CASSCF (CD-SA-CASSCF)
  - Cholesky Decomposition-State Average-Complete Active Space Perturbation Theory 2nd order (CD-SA-CASPT2)
  - Cholesky Decomposition-State Average-Complete Active Space Perturbation Theory 3rd order (CD-SA-CASPT3)
    
- **Configuration Interaction (CI)** (in progress)
  - Configuration Interaction Singles (CIS)
  - Configuration Interaction Singles and Doubles (CISD)
  - Configuration Interaction Singles, Doubles, and Triples (CISDT)
  - Full Configuration Interaction (FCI)
  - Multireference CI (MRCI)
  - CIPSI (Configuration Interaction by Perturbation with Selection Iteratively)


### Advanced Features
- **Integral Transformations**
  - Cholesky decomposition for electron repulsion integrals (ERI)
  - Electron Repulsion Integral Four-Index Transformation
  
- **Gradient and Optimization**
  - Analytical gradients for SCF methods
  - Numerical gradients
  - Geometry optimization

- **Analysis Tools**
  - Natural orbitals
  - Transition density matrices
  - Wavefunction analysis
  - One-particle density matrices (OPDM)

## 📋 Prerequisites

### Required Dependencies
- **C++ Compiler**: GCC 7+ or Clang 5+ (C++17 support)
- **CMake**: 3.15 or higher
- **Python**: 3.8 or higher
- **Libint2**: For integral evaluation
- **Eigen3**: For linear algebra operations
- **pybind11**: For Python bindings
- **NumPy**: For Python interface

### Optional Dependencies
- **OpenMP**: For parallel computation
- **MKL/OpenBLAS**: For optimized linear algebra


##  Installation Binding Python
## Using Conda (Cross-Platform)

```bash
# Create conda environment
conda create -n mshqc python=3.11
conda activate mshqc

# Install dependencies via conda-forge
conda install -c conda-forge \
    cmake \
    eigen \
    libint \
    numpy \
    pybind11

# Clone and install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install -e .
```
