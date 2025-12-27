#!/bin/bash

# ==============================================================================
# MSHQC Full Installer (Fixed Boost Dependency)
# Penulis: Muhamad Syahrul Hidayat
# ==============================================================================

set -e # Hentikan jika ada error

# Warna Output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ENV_NAME="mshqc"

echo -e "${YELLOW}======================================================${NC}"
echo -e "${YELLOW}   MSHQC COMPLETE INSTALLER (Reset Environment)       ${NC}"
echo -e "${YELLOW}======================================================${NC}"

# ------------------------------------------------------------------
# [0] PRE-FLIGHT CHECK
# ------------------------------------------------------------------
if [ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]; then
    echo -e "${RED}ERROR: Anda sedang berada di dalam environment '$ENV_NAME'.${NC}"
    echo "Mohon jalankan: 'conda deactivate' terlebih dahulu, lalu jalankan script ini lagi."
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda tidak ditemukan di sistem.${NC}"
    exit 1
fi

eval "$(conda shell.bash hook)"

# ------------------------------------------------------------------
# [1] RESET ENVIRONMENT CONDA
# ------------------------------------------------------------------
echo -e "\n${CYAN}[1/6] Menghapus environment lama '$ENV_NAME' (jika ada)...${NC}"
conda env remove -n $ENV_NAME -y > /dev/null 2>&1 || true

echo -e "\n${CYAN}[2/6] Membuat environment baru (Python 3.11)...${NC}"
conda create -n $ENV_NAME python=3.11 -y

# ------------------------------------------------------------------
# [2] INSTALL DEPENDENCIES (Conda-Forge)
# ------------------------------------------------------------------
echo -e "\n${CYAN}[3/6] Menginstall Library C++ & Python Dependencies...${NC}"
echo "Menginstall: cmake, eigen, libint, boost, dll..."

# FIX: Menambahkan 'boost-cpp' karena libint membutuhkannya
conda install -n $ENV_NAME -c conda-forge \
    cmake \
    make \
    compilers \
    eigen \
    libint \
    numpy \
    pybind11 \
    scikit-build \
    boost-cpp \
    -y

# ------------------------------------------------------------------
# [3] PERSIAPAN BUILD
# ------------------------------------------------------------------
echo -e "\n${CYAN}[4/6] Membersihkan artifact build lama...${NC}"
rm -rf build/ dist/ *.egg-info mshqc.egg-info
find python/ -name "*.so" -type f -delete
find python/ -name "*.pyd" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# ------------------------------------------------------------------
# [4] BUILD & INSTALL (PIP)
# ------------------------------------------------------------------
echo -e "\n${CYAN}[5/6] Mengompilasi Source Code (pip install)...${NC}"

conda activate $ENV_NAME

echo "Environment aktif: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

export CXXFLAGS="-std=c++17 -fopenmp -O3"

# Jalankan instalasi
pip install -e . -v

if [ $? -ne 0 ]; then
    echo -e "\n${RED}Instalasi GAGAL! Cek log error di atas.${NC}"
    exit 1
fi

# ------------------------------------------------------------------
# [5] VERIFIKASI
# ------------------------------------------------------------------
echo -e "\n${CYAN}[6/6] Verifikasi Akhir...${NC}"

python -c "import sys; print(f'Python Version: {sys.version.split()[0]}'); import mshqc; print(f'MSHQC Location: {mshqc.__file__}'); mol = mshqc.Molecule(); print('Core C++ Module: OK')"

if [ $? -eq 0 ]; then
    echo -e "\n${YELLOW}======================================================${NC}"
    echo -e "${GREEN}   INSTALASI SELESAI & SUKSES!                        ${NC}"
    echo -e "${YELLOW}======================================================${NC}"
    echo -e "Sekarang jalankan perintah berikut untuk mulai bekerja:\n"
    echo -e "    ${GREEN}conda activate $ENV_NAME${NC}"
    echo -e ""
else
    echo -e "\n${RED}Verifikasi GAGAL.${NC}"
    exit 1
fi