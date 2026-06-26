#!/bin/bash
set -e

echo -e "\033[1;36m=================================================================\033[0m"
echo -e "\033[1;36m              MEMBUAT FILE .WHL LOKAL (DENGAN CCACHE)            \033[0m"
echo -e "\033[1;36m=================================================================\033[0m"

# Pastikan menggunakan ccache agar prosesnya instan
export CC="ccache $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="ccache $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CXXFLAGS="-std=c++17 -fopenmp -O3 -mavx2 -mfma -mtune=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -DNDEBUG"

echo -e "\033[1;33m[*] Merakit Python Wheel (.whl)...\033[0m"
# Perintah ini yang membungkus C++ dan Python menjadi file .whl
python setup.py bdist_wheel

echo -e "\033[1;32m[*] Selesai! File .whl Anda ada di dalam folder dist/\033[0m"