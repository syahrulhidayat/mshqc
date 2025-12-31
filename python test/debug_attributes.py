import sys
sys.path.append("/home/syahrul/mshqc/python")
from mshqc.utils import quick_calculation

print("=== DEBUGGING ATRIBUT ===")

# 1. Jalankan kalkulasi He (Pasti berhasil karena genap)
print("Menghitung Helium RHF...")
res = quick_calculation("He", basis="sto-3g", method="rhf")

# 2. Intip isi objek hasil (res)
print("\n[DAFTAR ATRIBUT OBJEK HASIL]")
print(dir(res))

# 3. Coba akses manual (Mencoba menebak nama umum)
print("\n[PENGECEKAN NILAI]")
candidates = ['energy_total', 'e_tot', 'energy']
found = False
for name in candidates:
    if hasattr(res, name):
        val = getattr(res, name)
        print(f"✅ Ditemukan atribut '{name}': {val}")
        found = True

if not found:
    print("❌ Belum ketemu nama atribut energinya. Silakan kirim output daftar atribut di atas.")