#!/bin/bash

# ==============================================================================
# MSHQC Quick Rebuilder (Tanpa Install Ulang Library)
# Gunakan ini setelah mengubah kode C++ atau Python
# ==============================================================================

set -e

# Warna
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_NAME="mshqc"

echo -e "${YELLOW}[1/3] Cek Environment...${NC}"

# Cek apakah conda env sudah ada
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo -e "${RED}Error: Environment '$ENV_NAME' belum ada!${NC}"
    echo "Jalankan ./install.sh dulu untuk setup awal."
    exit 1
fi

# Aktifkan conda di shell script
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Environment aktif: $CONDA_DEFAULT_ENV"

# Bersihkan build C++ saja (biar yakin ter-update)
echo -e "${YELLOW}[2/3] Membersihkan cache build C++...${NC}"
# Kita HANYA hapus folder build cmake, jangan hapus dependencies lain
rm -rf build/
find python/ -name "*.so" -type f -delete

echo -e "${YELLOW}[3/3] Mengompilasi Ulang (Incremental Build)...${NC}"

# Setting Compiler
export CXXFLAGS="-std=c++17 -fopenmp -O3"

# Install dengan flag --no-deps (JANGAN cek dependensi library lagi, biar cepat)
# Flag --no-build-isolation mempercepat proses pip
pip install -e . -v --no-deps --no-build-isolation

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}SUCCESS! Kode berhasil di-update.${NC}"
else
    echo -e "\n${RED}Build GAGAL.${NC}"
    exit 1
fi