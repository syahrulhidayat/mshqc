#!/usr/bin/env python3
"""
Psi4 OMP2: Li atom doublet with cc-pVTZ basis
Compare with MSH-QC OMP2 implementation
"""

import psi4

psi4.set_memory('2 GB')
psi4.core.set_output_file('li_omp2_psi4.out', False)

# Li atom geometry (doublet, 2S+1=2)
li = psi4.geometry("""
0 2
Li 0.0 0.0 0.0
units angstrom
symmetry c1
""")

psi4.set_options({
    'basis': 'cc-pvtz',
    'reference': 'rohf',
    'scf_type': 'pk',
    'freeze_core': 'false',
    'e_convergence': 1e-10,
    'd_convergence': 1e-8,
    'print': 2,
    'occ_tolerance': 1e-6,  # OCC convergence
})

# Run ROHF first
print("\n" + "="*70)
print("ROHF Calculation")
print("="*70)
e_rohf = psi4.energy('scf')
print(f"\nROHF Energy: {e_rohf:.10f} Ha")

# Run OMP2
print("\n" + "="*70)
print("OMP2 Calculation")
print("="*70)
e_omp2 = psi4.energy('omp2')

# Extract correlation
omp2_vars = psi4.core.variables()
e_corr = omp2_vars.get('OMP2 CORRELATION ENERGY')
e_ss = omp2_vars.get('OMP2 SAME-SPIN CORRELATION ENERGY')
e_os = omp2_vars.get('OMP2 OPPOSITE-SPIN CORRELATION ENERGY')

print("\n" + "="*70)
print("OMP2 RESULTS (Psi4 Reference)")
print("="*70)
print(f"Reference ROHF energy:     {e_rohf:.10f} Ha")
print(f"Same-spin correlation:     {e_ss:.10f} Ha")
print(f"Opposite-spin correlation: {e_os:.10f} Ha")
print(f"Total OMP2 correlation:    {e_corr:.10f} Ha")
print(f"Total OMP2 energy:         {e_omp2:.10f} Ha")
print("="*70)

# Comparison with MSH-QC
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
mshqc_rohf = -7.431221
mshqc_omp2_corr = -0.608207
mshqc_omp2_total = -8.039427

print(f"\nROHF Energy:")
print(f"  Psi4:   {e_rohf:.10f} Ha")
print(f"  MSH-QC: {mshqc_rohf:.10f} Ha")
print(f"  Δ:      {abs(e_rohf - mshqc_rohf):.2e} Ha")

print(f"\nOMP2 Correlation:")
print(f"  Psi4:   {e_corr:.10f} Ha")
print(f"  MSH-QC: {mshqc_omp2_corr:.10f} Ha")
print(f"  Δ:      {abs(e_corr - mshqc_omp2_corr):.2e} Ha")

print(f"\nOMP2 Total:")
print(f"  Psi4:   {e_omp2:.10f} Ha")
print(f"  MSH-QC: {mshqc_omp2_total:.10f} Ha")
print(f"  Δ:      {abs(e_omp2 - mshqc_omp2_total):.2e} Ha")
print("="*70)
