# 1. Hapus cache indeks pelacakan Git lokal secara aman
git rm -r --cached .

# 2. Indeks ulang semua berkas berdasarkan aturan .gitignore baru
git add .

# 3. Komit perubahan aturan tersebut
git commit -m "chore: optimize .gitignore rules for quantum chemistry project"

# 4. Dorong ke GitHub
git push origin mshqc-1.0.0
