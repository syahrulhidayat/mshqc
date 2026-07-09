import psi4

psi4.set_memory('2 GB')
psi4.set_num_threads(4)

# Paksa Psi4 mencetak semuanya langsung ke layar terminal
psi4.core.set_output_file('stdout', False)

# Geometri H2O
ANG_TO_BOHR = 1.88972612462577 
mol_str = f"""
0 1
units bohr
O 0.0 0.0 {0.1173 * ANG_TO_BOHR:.10f}
H 0.0 {0.7572 * ANG_TO_BOHR:.10f} {-0.4692 * ANG_TO_BOHR:.10f}
H 0.0 {-0.7572 * ANG_TO_BOHR:.10f} {-0.4692 * ANG_TO_BOHR:.10f}
"""
psi4.geometry(mol_str)

# Opsi Super Cerewet (Print = 3)
psi4.set_options({
    'basis': 'cc-pVTZ',
    'df_basis_scf': 'cc-pVTZ-RI',
    'df_basis_mp2': 'cc-pVTZ-RI',
    'reference': 'rhf',
    'scf_type': 'df',
    'mp2_type': 'df',
    'e_convergence': 1e-9,
    'd_convergence': 1e-9,
    'freeze_core': 'False',
    'puream': True,
    'print': 3  # Level print tertinggi agar OMP2 keluar
})

print("="*60)
print(" MEMULAI PSI4 OMP2 RAW DUMP...")
print("="*60)

try:
    psi4.energy('omp2')
except Exception as e:
    print("Error:", e)