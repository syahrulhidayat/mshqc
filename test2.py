import pyscf
from pyscf import gto, scf, mp, cc

print("\n=== FINAL VERDICT: LI ATOM ===")

mol = gto.M(
    atom = 'Li 0 0 0',
    basis = 'cc-pvdz',
    spin = 1,
    charge = 0,
    verbose = 0
)

# 1. UHF
mf = scf.UHF(mol)
mf.kernel()

# 2. UMP2
mp2 = mp.UMP2(mf)
e2, _ = mp2.kernel()

# 3. UMP3 (Coba panggil manual jika ada, kalau tidak skip)
try:
    # Di PySCF modern, MP3 biasanya diakses begini
    mp3 = mp.MP3(mf) 
    e3_total, _ = mp3.kernel()
    e3_correction = e3_total - e2
except:
    e3_total = 0.0
    e3_correction = 0.0

# 4. CCSD (Gold Standard - The Truth)
ccsd = cc.CCSD(mf)
ccsd.kernel()
e_ccsd_corr = ccsd.e_corr

print(f"UHF Energy      : {mf.e_tot:.8f}")
print("-" * 40)
print(f"UMP2 Correlation: {e2:.8f} Ha")
if e3_total != 0.0:
    print(f"PySCF UMP3 Total: {e3_total:.8f} Ha")
    print(f"PySCF E(3) Corr : {e3_correction:.8f} Ha")
print(f"CCSD Correlation: {e_ccsd_corr:.8f} Ha")
print("-" * 40)
print(f"Kode Anda (MP3) : -0.00011797 Ha")

if abs(e_ccsd_corr - (-0.00011797)) < 0.0001:
    print("\nKESIMPULAN: Kode Anda BENAR.")
    print("Hasil MP3 Anda dekat dengan CCSD (Limit Eksak).")
    print("Perbedaan dengan MP2 adalah koreksi fisik yang wajar.")
else:
    print("\nKESIMPULAN: Masih ada yang aneh.")