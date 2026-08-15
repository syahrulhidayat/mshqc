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

import psi4

# 1. Alokasi Memori dan Thread
psi4.set_memory('4 GB')
psi4.set_num_threads(4)

# 2. Atur agar Output langsung dicetak ke Terminal/Stdout
# Ini akan menampilkan log iterasi secara real-time saat skrip dijalankan
psi4.core.set_output_file('stdout', False)

# 3. Definisikan Geometri H2O (Sama persis dengan benchmark-mu)
ANG_TO_BOHR = 1.88972612462577
mol_str = f"""
0 1
units bohr
O 0.0 0.0 {0.1173 * ANG_TO_BOHR:.10f}
H 0.0 {0.7572 * ANG_TO_BOHR:.10f} {-0.4692 * ANG_TO_BOHR:.10f}
H 0.0 {-0.7572 * ANG_TO_BOHR:.10f} {-0.4692 * ANG_TO_BOHR:.10f}
"""
psi4.geometry(mol_str)

# 4. Konfigurasi Opsi Psi4 untuk Menampilkan Detail Iterasi
psi4.set_options({
    'basis': 'cc-pVTZ',
    'df_basis_scf': 'cc-pVTZ-RI',
    'df_basis_mp2': 'cc-pVTZ-RI',
    'scf_type': 'df',
    'mp2_type': 'df',
    'freeze_core': 'False',
    'puream': True,
    
    # --- CRITICAL FOR DEBUGGING ---
    'print': 2,                 # Menaikkan level print global Psi4
    'scf_print': 2,             # Menampilkan tabel iterasi DIIS, energi, dan delta densitas RHF
    'e_convergence': 1e-9,      # Threshold energi (seperti mshqc)
    'd_convergence': 1e-9,      # Threshold density matrix
})

print("\n" + "="*60)
print(" JALANKAN KALKULASI 1: RHF (RESTRICTED HARTREE-FOCK)")
print("="*60)

# Menjalankan HF saja dulu untuk melihat iterasinya
hf_energy = psi4.energy('scf')

print("\n" + "="*60)
print(" JALANKAN KALKULASI 2: OMP2 (ORBITAL-OPTIMIZED MP2)")
print("="*60)

# Menjalankan OMP2 untuk melihat bagaimana SOSCF/Trust-Region bekerja di Psi4
omp2_energy = psi4.energy('omp2')

print("\n" + "="*60)
print(" RINGKASAN ENERGI AKHIR")
print("="*60)
print(f"RHF Energy  : {hf_energy:.12f} Ha")
print(f"OMP2 Energy : {omp2_energy:.12f} Ha")
print("="*60)