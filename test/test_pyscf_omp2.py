 # ==============================================================================
 # Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================

import time
from pyscf import gto, scf
from pyscf.mp import oomp2 # Impor modul OOMP2

# 1. Konfigurasi Molekul H2O
mol = gto.M(
    atom='''
    O   0.0000000000   0.0000000000   0.1173000000
    H   0.0000000000   0.7572000000  -0.4692000000
    H   0.0000000000  -0.7572000000  -0.4692000000
    ''',
    unit='angstrom',
    basis='cc-pvtz',
    symmetry=False,
    verbose=4
)

# 2. Hartree-Fock dengan Density Fitting (DF-RHF)
print("=== Memulai DF-RHF ===")
mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = 'cc-pvtz-ri' 
mf.conv_tol = 1e-10
mf.kernel()

# 3. Orbital-Optimized MP2 dengan Density Fitting (DF-OOMP2)
print("\n=== Memulai DF-OOMP2 ===")
t0 = time.time()

# PySCF OOMP2 otomatis mendeteksi jika 'mf' menggunakan DF
pt = oomp2.OOMP2(mf)
pt.frozen = 0    # All-Electron
pt.conv_tol = 1e-9
pt.kernel()

t1 = time.time()

# 4. Laporan Hasil
print("\n========================================================")
print("              LAPORAN PERFORMA PySCF DF-OOMP2           ")
print("========================================================")
print(f"E_SCF       : {mf.e_tot:.10f} Ha")
print(f"E_corr OOMP2: {pt.e_corr:.10f} Ha")
print(f"E_tot OOMP2 : {pt.e_tot:.10f} Ha")
print(f"Waktu OOMP2 : {t1 - t0:.4f} s")
print("========================================================")