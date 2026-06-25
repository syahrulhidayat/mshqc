#!/bin/bash
set -e

echo -e "\033[1;36m=================================================================\033[0m"
echo -e "\033[1;36m           MSHQC LOKAL TURBO BUILD (DEVELOPER MODE)              \033[0m"
echo -e "\033[1;36m=================================================================\033[0m"

# Pastikan ccache digunakan sebagai compiler utama
export CC="ccache $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="ccache $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CXXFLAGS="-std=c++17 -fopenmp -O3 -mavx2 -mfma -mtune=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -DNDEBUG"

echo -e "\033[1;33m[*] Mengompilasi C++ (Hanya file yang berubah)...\033[0m"
# build_ext --inplace adalah kunci agar kompilasi langsung jadi file .so di tempat, tanpa membuat .whl
python setup.py build_ext --inplace

echo -e "\033[1;32m[*] MSHQC Lokal Siap! Ccache menyimpan waktu Anda.\033[0m"