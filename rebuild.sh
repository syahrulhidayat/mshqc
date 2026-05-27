#!/bin/bash
set -e

echo -e "\033[1;36m=================================================================\033[0m"
echo -e "\033[1;36m           MSHQC TURBO BUILDER (GCC 15.2 + AUTO-TUNING)          \033[0m"
echo -e "\033[1;36m=================================================================\033[0m"

# 1. AKTIFKAN ENVIRONMENT PYTHON (Conda) TERLEBIH DAHULU!
ENV_NAME="psi4env"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 2. PAKSA MENGGUNAKAN KOMPILER SISTEM (DI BAWAH CONDA ACTIVATE)
# Conda tidak akan bisa menimpa instruksi ini karena diletakkan setelah activate!
export CC=/usr/bin/cc
export CXX=/usr/bin/c++

# 3. PARAMETER OPTIMASI HARDWARE TINGKAT EKSTREM
export CXXFLAGS="-std=c++17 -fopenmp -O3 -march=native -mtune=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE"

echo -e "\033[1;33m[1/4] Membersihkan Cache dan Biner Lama...\033[0m"
rm -rf build/
rm -f python/mshqc/libmshqc.so
rm -rf python/mshqc/*.so

echo -e "\033[1;33m[2/4] Memulai Kompilasi Murni dengan GCC 15...\033[0m"
python setup.py build_ext --inplace

# 4. PENYELAMAT MODUL PYTHON
echo -e "\033[1;33m[3/4] Mendaftarkan Modul ke Environment Python...\033[0m"
python -m pip install -e . > /dev/null 2>&1

echo -e "\033[1;32m[4/4] SUCCESS! MSHQC siap dieksekusi.\033[0m"
echo -e "\033[1;36m=================================================================\033[0m"
echo -e "Silakan jalankan pengujian Anda:"
echo -e "python \"test/pyhton test baru/test_mp2.py\""
echo -e "\033[1;36m=================================================================\033[0m"