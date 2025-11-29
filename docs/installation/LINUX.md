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
