#!/bin/bash
set -e

echo -e "\033[1;36m=================================================================\033[0m"
echo -e "\033[1;36m           MSHQC TURBO BUILDER (GCC 15.2 + AUTO-TUNING)          \033[0m"
echo -e "\033[1;36m=================================================================\033[0m"

# ==============================================================================
# JALUR PENYELAMAT: AMBIL PATH CONDA SECARA MANUWAL JIKA TERMINAL ERROR
# ==============================================================================
export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Jika cara di atas gagal, fallback ke hook bawaan Anda
    eval "$(conda shell.bash hook)"
fi

# AKTIFKAN ENVIRONMENT YANG BENAR
ENV_NAME="psi4env"
conda activate $ENV_NAME

# ==============================================================================
# PAKSA MENGGUNAKAN KOMPILER SISTEM
# ==============================================================================
export CC=/usr/bin/cc
export CXX=/usr/bin/c++

# PARAMETER OPTIMASI HARDWARE TINGKAT EKSTREM
export CXXFLAGS="-std=c++17 -fopenmp -O3 -march=native -mtune=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE"

echo -e "\033[1;33m[2/4] Memulai Kompilasi Murni dengan GCC 15...\033[0m"
python setup.py build_ext --inplace

echo -e "\033[1;32m[4/4] SUCCESS! MSHQC siap dieksekusi.\033[0m"
echo -e "\033[1;36m=================================================================\033[0m"
echo -e "Silakan jalankan pengujian Anda:"
echo -e "python \"test/pyhton test baru/test_mp2.py\""
echo -e "\033[1;36m=================================================================\033[0m"