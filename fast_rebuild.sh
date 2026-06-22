#!/bin/bash
set -e

echo -e "\033[1;36m=================================================================\033[0m"
echo -e "\033[1;36m           MSHQC TURBO BUILDER (BYPASS PIP ISOLATION)            \033[0m"
echo -e "\033[1;36m=================================================================\033[0m"

# 1. AKTIFASI ENVIRONMENT CONDA
source /opt/conda/etc/profile.d/conda.sh
conda activate psi4env

export CC=/usr/bin/cc
export CXX=/usr/bin/c++
export CXXFLAGS="-std=c++17 -fopenmp -O3 -march=native -mtune=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE"
export PYTHONWARNINGS="ignore"

echo -e "\033[1;33m[1/4] Membersihkan Cache dan Biner Lama...\033[0m"
rm -rf build/
rm -f python/mshqc/*.so
rm -f *.so

echo -e "\033[1;33m[2/4] Kompilasi C++ Murni via setup.py...\033[0m"
# Menggunakan kompilasi langsung agar CMake dapat melihat isi penuh psi4env
python setup.py build_ext --inplace

echo -e "\033[1;33m[3/4] Memastikan Posisi Biner Ekstensi Tepat Sasaran...\033[0m"
# Memaksa file _mshqc.so (jika tersasar di root atau folder build) masuk ke python/mshqc/
mv *.so python/mshqc/ 2>/dev/null || true
mv build/lib*/*.so python/mshqc/ 2>/dev/null || true

echo -e "\033[1;33m[4/4] Menyuntikkan Modul ke Core Python Conda...\033[0m"
# Mendapatkan jalur instalasi pustaka bawaan Conda
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Membuat fail .pth yang secara permanen mengarahkan Python ke folder kerja kita
echo "/workspaces/mshqc/python" > "$SITE_PACKAGES/mshqc.pth"

# Menyuntikkan LD_PRELOAD secara global di terminal agar TLS Jemalloc tidak bocor
export LD_PRELOAD="/lib/x86_64-linux-gnu/libjemalloc.so.2"

echo -e "\033[1;32mSUCCESS! MSHQC tertanam secara permanen di psi4env.\033[0m"
echo -e "\033[1;36m=================================================================\033[0m"
echo -e "CARA EKSEKUSI YANG BENAR (WAJIB PAKAI CONDA):"
echo -e "conda run -n psi4env python \"python test/test_mp2(1).py\""
echo -e "\033[1;36m=================================================================\033[0m"