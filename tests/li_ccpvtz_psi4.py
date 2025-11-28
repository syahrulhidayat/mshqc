#!/usr/bin/env python3
"""
Psi4 input: Li atom doublet with cc-pVTZ basis
Compare ROHF-MP2 with MSH-QC implementation
"""

import psi4

psi4.set_memory('2 GB')
psi4.core.set_output_file('li_ccpvtz_psi4.out', False)

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
    'mp2_type': 'conv',  # Conventional MP2 (not DF)
    'freeze_core': 'false',  # ALL electrons active!
    'e_convergence': 1e-10,
    'd_convergence': 1e-10,
    'print': 2
})

# Run ROHF
print("\n" + "="*70)
print("ROHF Calculation")
print("="*70)
e_rohf = psi4.energy('scf')
print(f"\nROHF Energy: {e_rohf:.10f} Ha")

# Run ROHF-MP2
print("\n" + "="*70)
print("ROHF-MP2 Calculation")
print("="*70)
e_mp2 = psi4.energy('mp2')

# Extract correlation components
mp2_vars = psi4.core.variables()
e_corr = mp2_vars.get('MP2 CORRELATION ENERGY')
e_ss = mp2_vars.get('MP2 SAME-SPIN CORRELATION ENERGY')
e_os = mp2_vars.get('MP2 OPPOSITE-SPIN CORRELATION ENERGY')

print("\n" + "="*70)
print("ROHF-MP2 RESULTS (Psi4)")
print("="*70)
print(f"Reference ROHF energy:     {e_rohf:.10f} Ha")
print(f"Same-spin correlation:     {e_ss:.10f} Ha")
print(f"Opposite-spin correlation: {e_os:.10f} Ha")
print(f"Total MP2 correlation:     {e_corr:.10f} Ha")
print(f"Total ROHF-MP2 energy:     {e_mp2:.10f} Ha")
print("="*70)

# Additional info
print("\n" + "="*70)
print("Orbital Information")
print("="*70)
wfn = psi4.core.get_active_wavefunction()
print(f"Alpha electrons: {wfn.nalpha()}")
print(f"Beta electrons:  {wfn.nbeta()}")
print(f"Multiplicity:    {wfn.multiplicity()}")
print("="*70)
