#!/bin/bash

# Pastikan eksekusi selalu dari root folder mshqc
cd "$(dirname "$0")"

echo "=========================================================="
echo "  MSHQC GitHub Push Script (Sinkronisasi & Pembersihan)"
echo "=========================================================="

# 1. Menghapus folder yang dikecualikan dari Git Cache (tanpa menghapus file aslinya di lokal)
# Angka 2>/dev/null digunakan agar tidak muncul error jika file belum masuk cache
echo "[1/4] Membersihkan cache git untuk folder yang diabaikan..."
git rm -r --cached build/ "python test/" hasil/ 2>/dev/null

# 2. Menambahkan semua perubahan (File baru, file dimodifikasi, dan file yang DIHAPUS)
echo "[2/4] Merekam perubahan struktur dan file terhapus..."
git add -A

# 3. Menangkap argumen pertama sebagai pesan commit (jika kosong, gunakan default)
COMMIT_MSG=$1
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update arsitektur mshqc, sinkronisasi file terhapus, dan optimasi struktur"
fi

echo "[3/4] Melakukan commit dengan pesan: '$COMMIT_MSG'"
git commit -m "$COMMIT_MSG"

# 4. Melakukan push ke origin branch main (ubah 'main' jika Anda menggunakan 'master')
echo "[4/4] Mengunggah ke https://github.com/syahrulhidayat/mshqc.git..."
git push origin main

echo "=========================================================="
echo "  Push Selesai!"
echo "=========================================================="