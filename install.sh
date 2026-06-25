#!/bin/bash
set -e
echo -e "\033[1;36m=================================================================\033[0m"
echo -e "\033[1;36m               INSTALLER MSHQC (UNTUK PENGGUNA AWAM)             \033[0m"
echo -e "\033[1;36m=================================================================\033[0m"

# 1. Deteksi Instalasi Conda secara Pintar
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo -e "\033[1;31m[ERROR] Conda tidak ditemukan! Harap install Miniconda/Anaconda.\033[0m"
    exit 1
fi

# 2. Amankan Dependensi C++ & Kunci Versi HDF5
echo -e "\033[1;33m[1/4] Mengunduh dependensi cerdas ke psi4env...\033[0m"
conda install -y -n psi4env -c conda-forge python=3.12 cmake make compilers eigen pkg-config "hdf5=1.14.3" pip libcint tblis liblapacke openblas psi4
conda activate psi4env

# 3. SUNTIKAN RAHASIA: Auto-Link & Bypass Versi HDF5
echo -e "\033[1;33m[2/4] Menghubungkan memori sistem (Mencegah Error .so)...\033[0m"

# A. Agar OS tahu tempat Conda menyimpan library
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/mshqc_env.sh

# B. (JURUS NINJA) Membohongi Linux agar selalu menemukan file HDF5 yang diminta MSHQC
ln -sf $CONDA_PREFIX/lib/libhdf5_cpp.so $CONDA_PREFIX/lib/libhdf5_cpp.so.320
ln -sf $CONDA_PREFIX/lib/libhdf5.so $CONDA_PREFIX/lib/libhdf5.so.320
# 4. Tarik Pembaruan dari Cloud
echo -e "\033[1;33m[3/4] Menarik biner superpower dari server...\033[0m"
git pull origin main --no-rebase

# 5. Install Biner Python
echo -e "\033[1;33m[4/4] Menginstal MSHQC...\033[0m"
python -m pip install releases/*.whl --force-reinstall

echo -e "\033[1;32m=================================================================\033[0m"
echo -e "\033[1;32m INSTALASI SUKSES 100%! \033[0m"
echo -e " Silakan jalankan program dengan cara:"
echo -e " 1. Ketik: conda activate psi4env"
echo -e " 2. Ketik: python tests/test_mp2.py"
echo -e "\033[1;32m=================================================================\033[0m"