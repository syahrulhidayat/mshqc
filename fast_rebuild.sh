#!/bin/bash

# ==============================================================================
# MSHQC TURBO REBUILDER (Incremental Build Mode)
# ==============================================================================

set -e

# Warna
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_NAME="mshqc"

# 1. Cek Environment
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo -e "${YELLOW}[Info] Mengaktifkan environment $ENV_NAME...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
fi

# 2. JANGAN HAPUS FOLDER BUILD! (Bagian ini dikomentari agar incremental jalan)
# rm -rf build/ 
# find python/ -name "*.so" -type f -delete

echo -e "${YELLOW}[Target] Memperbarui file C++ yang berubah saja...${NC}"

# Setting Compiler
export CXXFLAGS="-std=c++17 -fopenmp -O3"

# 3. Compile Cepat
# Perintah ini akan mengecek folder 'build/'.
# Jika 'basis.cc' lebih baru dari 'basis.o', dia compile basis.cc.
# Jika file lain tidak berubah, dia SKIP (hemat waktu).
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}SUCCESS! Library berhasil di-update.${NC}"
else
    echo -e "\n${RED}Build GAGAL.${NC}"
    exit 1
fi