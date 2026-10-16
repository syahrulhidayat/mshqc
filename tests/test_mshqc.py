import mshqc

# 1. Initialize molecule (supports atomic symbols/numbers and "angstrom" or "bohr" units)
mol = mshqc.Molecule(charge=0, multiplicity=1)
mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

# 2. Load basis set and analytical integral engine
basis     = mshqc.BasisSet("cc-pVTZ", mol)
integrals = mshqc.IntegralEngine(mol, basis)

# 3. Run RHF with Cholesky-decomposed ERIs (threshold = 1e-9 Ha)
scf_conf = mshqc.SCFConfig(eri_method="exact", threshold=1e-9)
scf_res  = mshqc.RHF(mol, basis, integrals, scf_conf).compute()
print(f"RHF  Total Energy : {scf_res.energy_total:.10f} Ha")

# 4. Run RMP2, OMP2, and OMP3 on the converged SCF reference
mp2_conf = mshqc.MP2Config(eri_method="exact", threshold=1e-9, gradient_threshold=1e-5)

rmp2_res = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()

print(f"RMP2 Total Energy : {rmp2_res.energy_total:.10f} Ha (Corr: {rmp2_res.energy_mp2_corr:.10f} Ha)")
print(f"OMP2 Total Energy : {omp2_res.energy_total:.10f} Ha")
print(f"OMP3 Total Energy : {omp3_res.energy_total:.10f} Ha")