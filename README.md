# MSH-QC: Quantum Mechanics Library for Computational Chemistry

MSH-QC is a comprehensive quantum mechanics library designed for accurate molecular simulations. This C++ library provides a complete implementation of electronic structure methods from basic Hartree-Fock to advanced multi-reference approaches, enabling high-precision calculations of molecular properties and energetics.

## Features

### Self-Consistent Field (SCF) Methods
- RHF (Restricted Hartree-Fock)
- ROHF (Restricted Open-shell Hartree-Fock) 
- UHF (Unrestricted Hartree-Fock)

### Møller-Plesset Perturbation Theory
- RMP2, UMP2, OMP2
- RMP3, UMP3, OMP3
- UMP4, UMP5 (higher-order perturbation theory)
- Density-Fitting MP2 (DF-MP2)

### Configuration Interaction (CI) Methods
- CIS (Configuration Interaction Singles)
- CISD (Configuration Interaction Singles and Doubles)
- CISDT (Configuration Interaction Singles, Doubles, and Triples)
- FCI (Full Configuration Interaction)
- MRCI (Multi-Reference Configuration Interaction)
- CIPSI (Selected CI with Perturbative Selection)

### Multi-Configurational Methods
- CASSCF (Complete Active Space SCF)
- SA-CASSCF (State-Averaged CASSCF)
- CASPT2 (Complete Active Space Perturbation Theory)
- DF-CASPT2 (Density-Fitting CASPT2)
- MRMP2 (Multi-Reference Møller-Plesset Perturbation Theory)

### Analysis Tools
- Natural Orbital Analysis
- Wavefunction Analysis
- One- and Two-Particle Density Matrices
- Numerical Gradients
- Geometry Optimization

## Dependencies

### Required
- C++17 compiler (GCC 9+)
- CMake 3.14+
- Eigen 3.3+
- Libint2 2.7+

### Optional
- OpenMP (parallelization)
- MKL/BLAS (faster linear algebra)
- Libcint (for 3-center integrals and density fitting)

## Installation

### Option 1: Quick Install with Script
Run the installation script for automated dependency installation:

```bash
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
chmod +x install.sh
./install.sh
```

### Option 2: Manual Installation

#### Dependencies
- C++17 compiler (GCC 9+ or Clang 8+)
- CMake 3.14+
- Eigen 3.3+
- Libint2 2.7+ (for electron repulsion integrals)
- BLAS/LAPACK (for linear algebra)
- Libcint (optional, for density fitting)
- OpenMP (optional, for parallelization)

#### Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake libeigen3-dev libblas-dev liblapack-dev libint2-dev libomp-dev
```

**Fedora/CentOS:**
```bash
sudo dnf install gcc gcc-c++ cmake eigen3-devel blas-devel lapack-devel libint2-devel libgomp-devel
```

**macOS (with Homebrew):**
```bash
brew install cmake eigen libint2 libomp
```

#### Build and Install MSH-QC
```bash
# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install
sudo make install
```

## Using MSH-QC in Your Project

After installation, you can use MSH-QC in your own CMake project:

```cmake
# In your CMakeLists.txt
find_package(MSHQC REQUIRED)
target_link_libraries(your_target mshqc)
```

## Building

## Python Installation

### Option 1: pip install from GitHub
```bash
pip install git+https://github.com/syahrulhidayat/mshqc.git
```

### Option 2: Build from source
```bash
# Clone the repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install Python dependencies
pip install numpy pybind11

# Install the Python package
pip install -e .
```

### Python Example
```python
import mshqc

# Create H2 molecule
mol = mshqc.create_h2_molecule()

# Perform RHF calculation
result = mshqc.quick_rhf(mol, "sto-3g")
print(f"RHF Energy: {result.energy:.6f} Hartree")

# Perform MP2 calculation
rhf_result, mp2 = mshqc.quick_mp2(mol, "sto-3g")
print(f"MP2 Energy: {mp2.get_energy():.6f} Hartree")
```

## Testing

## License

MIT License

## Citation

If you use this package in your research, please cite:

```
MSH-QC: Minimalist Scalable High-Performance Quantum Chemistry
[Add citation information when published]
```