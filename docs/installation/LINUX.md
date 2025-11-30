# Linux Installation Guide

## Ubuntu/Debian

### Instalasi Lengkap

```bash
# Update sistem
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    libeigen3-dev \
    libint2-dev

# Buat virtual environment
python3 -m venv ~/mshqc-env
source ~/mshqc-env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy pybind11

# Clone dan install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Install dengan semua fitur
export MSHQC_WITH_LIBINT2=ON
export MSHQC_WITH_LIBCINT=ON
pip install .
```

### Instalasi Minimal

```bash
# Dependencies minimal
sudo apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    libeigen3-dev

# Setup environment
python3 -m venv ~/mshqc-env
source ~/mshqc-env/bin/activate
pip install numpy pybind11

# Install tanpa fitur eksternal
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

export MSHQC_WITH_LIBINT2=OFF
export MSHQC_WITH_LIBCINT=OFF
pip install .
```

## Fedora/RHEL/CentOS

```bash
# Install dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    git \
    eigen3-devel \
    python3-devel \
    python3-pip

# Setup dan install
python3 -m venv ~/mshqc-env
source ~/mshqc-env/bin/activate
pip install numpy pybind11

git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```
# Installing MSHQC on Fedora Linux

This guide covers installation of MSHQC on Fedora, RHEL, CentOS, and Rocky Linux.

## Prerequisites

### System Requirements
- Fedora 36+ / RHEL 8+ / CentOS Stream 8+
- 4 GB RAM minimum (8 GB recommended)
- 2 GB free disk space
- Internet connection for downloading packages

### Required Packages
- GCC 7+ with C++17 support
- CMake 3.15+
- Python 3.8+
- Eigen3 library
- Libint2 library

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Update system
sudo dnf update -y

# Install build tools and dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    make \
    git \
    eigen3-devel \
    python3-devel \
    python3-pip

# Install Python packages
pip3 install --user numpy pybind11

# Clone and build MSHQC
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Run automated installation
bash install_dependencies.sh
bash install.sh
```

### Method 2: Manual Installation

#### Step 1: Install System Dependencies

```bash
# Enable PowerTools/CRB repository (for RHEL/CentOS)
# For RHEL 8/CentOS Stream 8:
sudo dnf config-manager --set-enabled powertools
# For RHEL 9/CentOS Stream 9:
sudo dnf config-manager --set-enabled crb

# Install build essentials
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    gcc \
    gcc-c++ \
    cmake \
    make \
    git \
    wget \
    curl
```

#### Step 2: Install Eigen3

```bash
# Install from repository
sudo dnf install -y eigen3-devel

# Verify installation
pkg-config --modversion eigen3
```

If Eigen3 is not available in your repository:

```bash
# Download and install manually
cd /tmp
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install
```

#### Step 3: Install Libint2

Libint2 is often not available in Fedora repositories, so we'll build from source:

```bash
# Install Libint2 dependencies
sudo dnf install -y \
    boost-devel \
    gmp-devel \
    autoconf \
    automake \
    libtool

# Clone and build Libint2
cd /tmp
git clone https://github.com/evaleev/libint.git
cd libint

# Generate configure script
./autogen.sh

# Build and install
mkdir build && cd build
../configure --prefix=/usr/local \
    --enable-shared \
    --with-max-am=5 \
    --with-opt-am=3 \
    --enable-generic-code \
    --disable-unrolling

make -j$(nproc)
sudo make install

# Update library cache
sudo ldconfig
```

#### Step 4: Install Python Dependencies

```bash
# Install Python development packages
sudo dnf install -y \
    python3-devel \
    python3-pip \
    python3-setuptools \
    python3-wheel

# Upgrade pip
pip3 install --user --upgrade pip

# Install required Python packages
pip3 install --user numpy pybind11
```

#### Step 5: Build and Install MSHQC

```bash
# Clone repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    -DLibint2_DIR=/usr/local/lib/cmake/libint2

# Build
make -j$(nproc)

# Install (requires sudo)
sudo make install

# Or install to user directory
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install
```

#### Step 6: Install Python Bindings

```bash
# From the mshqc root directory
pip3 install --user -e .

# Or install globally (requires sudo)
sudo pip3 install .
```

### Method 3: Using Conda (Cross-Platform)

```bash
# Install Miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n mshqc python=3.11
conda activate mshqc

# Install dependencies
conda install -c conda-forge \
    cmake \
    eigen \
    libint \
    numpy \
    pybind11 \
    gcc_linux-64 \
    gxx_linux-64

# Clone and install
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install -e .
```

## Verification

### Test Installation

```bash
# Test C++ library
cd mshqc/build
ctest --output-on-failure

# Test Python bindings
python3 -c "import mshqc; print('MSHQC version:', mshqc.__version__)"

# Run example calculation
cd ../examples
cmake -B build
cmake --build build
./build/rhf_test
```

### Run Benchmark

```bash
cd mshqc/examples
./build/ci_tests/li_ccpvtz_test
```

## Troubleshooting

### Libint2 Not Found

If CMake cannot find Libint2:

```bash
# Set environment variable
export Libint2_DIR=/usr/local/lib/cmake/libint2

# Or specify in CMake
cmake .. -DLibint2_DIR=/usr/local/lib/cmake/libint2
```

### Eigen3 Not Found

```bash
# Install from repository
sudo dnf install eigen3-devel

# Or set path manually
export EIGEN3_INCLUDE_DIR=/usr/include/eigen3
cmake .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
```

### Python Module Not Found

```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Add installation path
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# Or reinstall
pip3 install --user --force-reinstall -e .
```

### Compilation Errors

```bash
# Ensure C++17 support
gcc --version  # Should be 7.0 or higher

# Update compiler if needed
sudo dnf install gcc-toolset-12
scl enable gcc-toolset-12 bash
```

### Library Path Issues

```bash
# Add library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Make permanent
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Or update ldconfig
echo '/usr/local/lib' | sudo tee /etc/ld.so.conf.d/local.conf
sudo ldconfig
```

### SELinux Issues

```bash
# Temporarily disable SELinux
sudo setenforce 0

# Or add exception
sudo setsebool -P allow_execheap 1
sudo setsebool -P allow_execmem 1
```

## Performance Optimization

### Use Intel MKL (Optional)

```bash
# Install MKL
sudo dnf install intel-mkl intel-mkl-devel

# Rebuild with MKL
cmake .. -DBLA_VENDOR=Intel10_64lp
make clean
make -j$(nproc)
```

### Enable OpenMP

```bash
# Install OpenMP
sudo dnf install libomp-devel

# CMake will automatically detect and enable
```

### Compiler Optimization Flags

```bash
# Build with native optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"
```

## Uninstallation

```bash
# Remove installed files
cd mshqc/build
sudo make uninstall

# Or if installed to user directory
rm -rf ~/.local/lib/python*/site-packages/mshqc*
rm -rf ~/.local/lib/libmshqc*
rm -rf ~/.local/include/mshqc

# Remove Conda environment
conda env remove -n mshqc
```

## Additional Resources

- [Fedora Docs](https://docs.fedoraproject.org/)
- [RHEL Documentation](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/)
- [MSHQC Issues](https://github.com/syahrulhidayat/mshqc/issues)

## Getting Help

If you encounter issues:

1. Check [Troubleshooting Guide](../TROUBLESHOOTING.md)
2. Search [existing issues](https://github.com/syahrulhidayat/mshqc/issues)
3. Open a new issue with:
   - Fedora/RHEL version: `cat /etc/fedora-release` or `cat /etc/redhat-release`
   - GCC version: `gcc --version`
   - CMake version: `cmake --version`
   - Error messages and logs
## Arch Linux

```bash
# Install dependencies
sudo pacman -S --needed \
    base-devel \
    cmake \
    eigen \
    python \
    python-pip \
    python-numpy

# Setup dan install
python -m venv ~/mshqc-env
source ~/mshqc-env/bin/activate
pip install pybind11

git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

## Verifikasi

```bash
python -c "import mshqc; print(mshqc.__version__)"
python scripts/check_installation.py
```
