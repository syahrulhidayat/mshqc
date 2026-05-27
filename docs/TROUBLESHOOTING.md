# Troubleshooting Guide

## Common Issues

### 1. "No module named 'mshqc'" setelah instalasi

**Solusi:**
```bash
# Cek apakah di environment yang benar
which python  # Linux/macOS
where python  # Windows

pip list | grep mshqc

# Reinstall jika perlu
pip uninstall mshqc
pip install --no-cache-dir .
```

### 2. Metode tertentu tidak tersedia

**Penyebab:** Library eksternal belum terinstall

**Solusi:**
```bash
# Linux - install dengan fitur lengkap
sudo apt-get install libint2-dev
export MSHQC_WITH_LIBINT2=ON
export MSHQC_WITH_LIBCINT=ON
pip install --force-reinstall .
```

### 3. Build error "Cannot find Eigen3"

**Solusi:**
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# Fedora
sudo dnf install eigen3-devel

# Arch
sudo pacman -S eigen

# Atau set path manual:
export EIGEN3_INCLUDE_DIR=/path/to/eigen3
```

### 4. Memory error saat kompilasi

**Solusi:**
```bash
# Limit parallel jobs
pip install . --install-option="--parallel=2"

# Atau dengan CMake:
cmake --build . -j2
```

### 5. ImportError dengan pybind11

**Solusi:**
```bash
pip install --upgrade pybind11
# Atau versi spesifik:
pip install pybind11==2.12.0
```

## Platform-Specific Issues

### Linux
- **Permission denied**: Gunakan virtual environment atau `--user` flag
- **Linking errors**: Jalankan `sudo ldconfig`

### Windows
- **MSVC error**: Install Visual Studio Community dengan C++ tools
- **PATH issues**: Restart terminal setelah install tools

### macOS
- **Compiler not found**: Install Xcode Command Line Tools
  ```bash
  xcode-select --install
  ```

## Getting Help

Jika masalah tetap ada:
1. [Buat Issue di GitHub](https://github.com/syahrulhidayat/mshqc/issues)
2. Sertakan:
   - OS dan versi
   - Python version
   - Output dari `check_installation.py`
   - Full error message
   - Build log jika ada