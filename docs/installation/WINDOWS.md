# Windows Installation Guide

## Metode 1: Visual Studio (Direkomendasikan)

### Prerequisites
1. Download dan install:
   - [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
   - Pilih "Desktop development with C++"
   - [CMake](https://cmake.org/download/)
   - [Python 3.8+](https://www.python.org/downloads/)
   - [Git](https://git-scm.com/download/win)

### Installation Steps

Buka "Developer Command Prompt for VS 2022":

```cmd
REM Buat virtual environment
python -m venv C:\mshqc-env
C:\mshqc-env\Scripts\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install numpy pybind11

REM Clone repository
cd C:\
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

REM Install (minimal - direkomendasikan untuk Windows)
set MSHQC_WITH_LIBINT2=OFF
set MSHQC_WITH_LIBCINT=OFF
pip install .
```

## Metode 2: MSYS2 (Lebih Mudah)

### Setup MSYS2
1. Download dari [msys2.org](https://www.msys2.org/)
2. Install ke `C:\msys64`
3. Jalankan "MSYS2 MINGW64"

### Installation

```bash
# Update packages
pacman -Syu

# Install dependencies
pacman -S --needed \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-eigen3 \
    mingw-w64-x86_64-python \
    mingw-w64-x86_64-python-pip \
    mingw-w64-x86_64-python-numpy \
    git

# Install MSH-QC
pip install pybind11
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install .
```

## Verifikasi

```cmd
python -c "import mshqc; print('MSH-QC installed successfully!')"
python scripts\check_installation.py
```

## Troubleshooting

**Error: "Microsoft Visual C++ 14.0 or greater is required"**
- Install Visual Studio dengan C++ development tools

**CMake not found**
- Pastikan CMake sudah ditambahkan ke PATH saat instalasi
- Atau restart terminal setelah instalasi CMake
