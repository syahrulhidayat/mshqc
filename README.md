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

- **Møller-Plesset Perturbation Theory**
  - MP2, MP3, MP4, MP5 (Restricted and Unrestricted)
  - Orbital-Optimized MP2/MP3 (OMP2/OMP3)
  - Density-Fitted MP2 (DF-MP2)

- **Configuration Interaction (CI)**
  - Configuration Interaction Singles (CIS)
  - Configuration Interaction Singles and Doubles (CISD)
  - Configuration Interaction Singles, Doubles, and Triples (CISDT)
  - Full Configuration Interaction (FCI)
  - Multireference CI (MRCI)
  - CIPSI (Configuration Interaction by Perturbation with Selection Iteratively)

- **Multi-Configurational Self-Consistent Field (MCSCF)**
  - Complete Active Space SCF (CASSCF)
  - State-Averaged CASSCF (SA-CASSCF)
  - Complete Active Space Perturbation Theory 2nd order (CASPT2)
  - Multireference MP2 (MRMP2)

### Advanced Features
- **Integral Transformations**
  - Cholesky decomposition for electron repulsion integrals (ERI)
  - Density-Fitted integrals
  - 3-center integral support
  
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

## 🚀 Installation

> **Note**: For detailed platform-specific instructions, see [Installation Guides](docs/installation/)

### Quick Install

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
# Clone the repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libint2-dev \
    python3-dev \
    python3-pip

# Install Python dependencies and build
pip install numpy pybind11
pip install -e .
```
</details>

<details>
<summary><b>Linux (Fedora/RHEL/CentOS)</b></summary>

```bash
# Clone the repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install system dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    eigen3-devel \
    libint2-devel \
    python3-devel \
    python3-pip

# If libint2-devel is not available, install from source
# See docs/installation/LINUX.md for details

# Install Python dependencies and build
pip install numpy pybind11
pip install -e .
```
</details>

<details>
<summary><b>Linux (Arch/Manjaro)</b></summary>

```bash
# Clone the repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install system dependencies
sudo pacman -S \
    base-devel \
    cmake \
    eigen \
    libint \
    python \
    python-pip

# Install Python dependencies and build
pip install numpy pybind11
pip install -e .
```
</details>

<details>
<summary><b>macOS (Homebrew)</b></summary>

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Clone the repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install dependencies
brew install cmake eigen libint python@3.11

# Install Python dependencies and build
pip3 install numpy pybind11
pip3 install -e .
```
</details>

<details>
<summary><b>Windows (WSL2 - Recommended)</b></summary>

```powershell
# Install WSL2 (in PowerShell as Administrator)
wsl --install -d Ubuntu-22.04

# Open WSL2 Ubuntu terminal, then follow Ubuntu instructions
# See above Ubuntu/Debian section
```
</details>

<details>
<summary><b>Windows (Native with MSYS2)</b></summary>

```bash
# Install MSYS2 from https://www.msys2.org/

# Open MSYS2 MinGW 64-bit terminal
pacman -S \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-eigen3 \
    mingw-w64-x86_64-python \
    mingw-w64-x86_64-python-pip

# Clone and install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Note: Libint2 must be built from source on Windows
# See docs/installation/WINDOWS.md for details

pip install numpy pybind11
pip install -e .
```
</details>

<details>
<summary><b>Windows (Visual Studio)</b></summary>

```powershell
# Prerequisites:
# - Visual Studio 2019 or later with C++ support
# - CMake (from cmake.org or via Visual Studio)
# - Python 3.8+ from python.org

# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install Python dependencies
pip install numpy pybind11

# Build with CMake
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# Install
cmake --install . --prefix C:/mshqc
```

**Note**: Libint2 and Eigen3 must be installed separately. See [Windows Installation Guide](docs/installation/WINDOWS.md)
</details>

### Using Conda (Cross-Platform)

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

### Using Installation Scripts (Linux/macOS)

```bash
# Install all dependencies automatically
bash install_dependencies.sh

# Build and install MSHQC
bash install.sh
```

### Manual Build with CMake (All Platforms)

```bash
# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Create build directory
mkdir build && cd build

# Configure (adjust paths as needed)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    -DLibint2_DIR=/usr/local/lib/cmake/libint2

# Build (use appropriate number of cores)
# Linux/macOS:
make -j$(nproc)
# Windows:
cmake --build . --config Release -j 8

# Install
sudo make install  # Linux/macOS
# or
cmake --install . --prefix C:/mshqc  # Windows
```

### Docker (All Platforms)

```bash
# Pull pre-built image (coming soon)
docker pull syahrulhidayat/mshqc:latest

# Or build from source
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
docker build -t mshqc .

# Run container
docker run -it mshqc
```

### Python Package Installation

```bash
# Development installation
pip install -e .

# Or from GitHub directly
pip install git+https://github.com/syahrulhidayat/mshqc.git

# Specify version
pip install git+https://github.com/syahrulhidayat/mshqc.git@v1.0.0
```

### Troubleshooting Installation

If you encounter issues during installation:

1. **Missing Libint2**: Most common issue. See platform-specific instructions above or build from source:
   ```bash
   git clone https://github.com/evaleev/libint.git
   cd libint
   ./autogen.sh
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
   make -j$(nproc)
   sudo make install
   ```

2. **Eigen3 not found**: Ensure Eigen3 is installed and CMake can find it:
   ```bash
   # Linux
   sudo apt-get install libeigen3-dev  # Ubuntu/Debian
   sudo dnf install eigen3-devel       # Fedora
   
   # macOS
   brew install eigen
   
   # Or download from https://eigen.tuxfamily.org/
   ```

3. **Python binding errors**: Ensure pybind11 is installed:
   ```bash
   pip install pybind11[global]
   ```

4. **Compiler errors**: Ensure you have a C++17 compatible compiler:
   - GCC 7+ (Linux)
   - Clang 5+ (macOS/Linux)
   - MSVC 2017+ (Windows)

For more detailed troubleshooting, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Quick Start Guide](docs/QUICKSTART.md)**: Get started quickly
- **[Installation Guides](docs/installation/)**
  - [Linux Installation](docs/installation/LINUX.md)
  - [Windows Installation](docs/installation/WINDOWS.md)
  - [Troubleshooting](docs/installation/TROUBLESHOOTING.md)
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## 🔧 Usage Examples

### Python Interface

```python
import mshqc
import numpy as np

# Define molecule (water)
mol = mshqc.Molecule()
mol.add_atom("O", [0.0, 0.0, 0.0])
mol.add_atom("H", [0.0, 0.757, 0.587])
mol.add_atom("H", [0.0, -0.757, 0.587])

# Set basis set
basis = mshqc.Basis("cc-pVDZ", mol)

# Run RHF calculation
rhf = mshqc.RHF(mol, basis)
rhf.run()
energy = rhf.get_energy()
print(f"RHF Energy: {energy:.8f} Hartree")

# Run MP2 calculation
mp2 = mshqc.MP2(rhf)
mp2.run()
mp2_energy = mp2.get_energy()
print(f"MP2 Energy: {mp2_energy:.8f} Hartree")

# Run CISD calculation
cisd = mshqc.CISD(rhf)
cisd.run()
cisd_energy = cisd.get_energy()
print(f"CISD Energy: {cisd_energy:.8f} Hartree")
```

### C++ Interface

```cpp
#include <mshqc/molecule.h>
#include <mshqc/basis.h>
#include <mshqc/scf.h>
#include <mshqc/mp2.h>

int main() {
    // Create molecule
    auto mol = std::make_shared<mshqc::Molecule>();
    mol->add_atom("H", 0.0, 0.0, 0.0);
    mol->add_atom("H", 0.0, 0.0, 1.4);
    
    // Initialize basis
    auto basis = std::make_shared<mshqc::Basis>("STO-3G", mol);
    
    // Run RHF
    mshqc::RHF rhf(mol, basis);
    rhf.compute();
    double e_rhf = rhf.energy();
    
    // Run MP2
    mshqc::MP2 mp2(rhf);
    mp2.compute();
    double e_mp2 = mp2.energy();
    
    std::cout << "RHF Energy: " << e_rhf << " Hartree\n";
    std::cout << "MP2 Energy: " << e_mp2 << " Hartree\n";
    
    return 0;
}
```

## 🧪 Testing

### Running Tests

```bash
# Build tests
cd build
cmake .. -DBUILD_TESTS=ON
make

# Run all tests
ctest --output-on-failure

# Or run specific test categories
./tests/test_scf
./tests/test_mp2
./tests/test_ci
```

### Example Tests

The `examples/` directory contains numerous test cases:

- **SCF Tests**: `rhf_test.cc`, `uhf_test.cc`, `rohf_test.cc`
- **MP Tests**: `mp_tests/rmp2_test.cc`, `mp_tests/ump3_test.cc`
- **CI Tests**: `ci_tests/cisd_h2_test.cc`, `ci_tests/fci_test.cc`
- **MCSCF Tests**: `mcscf_tests/casscf_test.cc`, `mcscf_tests/caspt2_test.cc`
- **Gradient Tests**: `gradient/test_gradient_h2o_pvdz.cc`

## 📊 Performance

MSHQC is designed for high performance:

- **Optimized Linear Algebra**: Uses Eigen3 with optional MKL/OpenBLAS backend
- **Memory Efficient**: Cholesky decomposition for large systems
- **Parallel Computing**: OpenMP support for multi-threading
- **Sparse Methods**: Efficient sparse matrix operations for CI

### Benchmarks

| Method | System | Basis | Time | Memory |
|--------|--------|-------|------|--------|
| RHF | H₂O | cc-pVDZ | 0.5s | 50 MB |
| MP2 | H₂O | cc-pVDZ | 1.2s | 100 MB |
| CISD | Li | cc-pVTZ | 3.5s | 200 MB |
| CASSCF(2,2) | H₂ | cc-pVDZ | 2.1s | 80 MB |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/mshqc.git
cd mshqc

# Create a development branch
git checkout -b feature/your-feature

# Make changes and test
# ...

# Submit a pull request
```

## 📖 Citation

If you use MSHQC in your research, please cite:

```bibtex
@software{mshqc2024,
  author = {Syahrul Hidayat},
  title = {MSHQC: Modern Quantum Chemistry Library},
  year = {2024},
  url = {https://github.com/syahrulhidayat/mshqc}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: https://github.com/syahrulhidayat/mshqc
- **Issues**: https://github.com/syahrulhidayat/mshqc/issues
- **Discussions**: https://github.com/syahrulhidayat/mshqc/discussions

## 👥 Authors

- **Muhamad Syahrul Hidayat** - *Main Developer*

## 🙏 Acknowledgments

- Libint2 project for integral evaluation
- Eigen library for linear algebra
- pybind11 for Python bindings
- The quantum chemistry community

## 📞 Support

If you encounter any issues:

1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Search existing [GitHub Issues](https://github.com/syahrulhidayat/mshqc/issues)
3. Create a new issue with detailed information

## 🗺️ Roadmap

### Upcoming Features
- [ ] Coupled Cluster methods (CCSD, CCSD(T))
- [ ] Time-dependent DFT (TD-DFT)
- [ ] Periodic boundary conditions
- [ ] GPU acceleration
- [ ] Extended basis set library
- [ ] Advanced visualization tools

---

**Last Updated**: November 2025
