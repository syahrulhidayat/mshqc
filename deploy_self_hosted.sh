#!/bin/bash
# Skrip injeksi I/O dan Remote Build untuk mshqc (Laptop -> Self-Hosted)
# Hentikan eksekusi jika terjadi error
set -e

# ==========================================
# Parameter Konfigurasi Remote
# ==========================================
REMOTE_USER="nama_user_server"
REMOTE_HOST="ip_atau_domain_server"
REMOTE_PORT="22"
REMOTE_DIR="/path/to/remote/mshqc_self_hosted"
CONDA_ENV="mshqc_314"

echo "[INFO] Memulai sinkronisasi I/O mshqc ke $REMOTE_HOST..."

# 1. Injeksi direktori secara presisi (mengabaikan artefak lokal yang terkompilasi)
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='releases/' \
    --exclude='*.egg-info' \
    --exclude='__pycache__/' \
    -e "ssh -p $REMOTE_PORT" \
    ./ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR

echo "[INFO] Sinkronisasi selesai. Memulai kompilasi jarak jauh..."

# 2. Eksekusi kompilasi via SSH (Headless)
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST << 'ENDSSH'
    set -e
    # Memuat profil conda
    source ~/miniconda3/etc/profile.d/conda.sh || source ~/.bashrc
    conda activate mshqc_314
    
    cd /path/to/remote/mshqc_self_hosted

    echo "[INFO] Memperbarui alokasi dependensi Conda (LLVM/MLIR)..."
    conda env update -f environment.yml --prune

    echo "[INFO] Menjalankan resolusi CMake dan generasi TableGen..."
    cmake -B build -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm -DMLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir
    cmake --build build --target MshqcCompilerIncGen -j $(nproc)

    echo "[INFO] Membangun Python Wheel untuk rilis publik..."
    # Hapus sisa build lama untuk memastikan clean state
    rm -rf build/lib.* dist/*
    python setup.py bdist_wheel
    
    echo "[INFO] Memindahkan artefak .whl ke direktori releases..."
    mkdir -p releases
    cp dist/*.whl releases/

    echo "[SUCCESS] Kompilasi selesai. Artefak berada di $REMOTE_DIR/releases/"
ENDSSH

# 3. Tarik (Pull) kembali artefak rilis ke laptop lokal
echo "[INFO] Menarik artefak rilis yang dikompilasi dari server..."
rsync -avz -e "ssh -p $REMOTE_PORT" $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/releases/ ./releases/

echo "[SUCCESS] Injeksi dan Kompilasi berhasil. Silakan cek direktori ./releases/"