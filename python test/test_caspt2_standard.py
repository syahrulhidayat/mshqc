import mshqc
import time

print("==========================================================")
print("   STANDARD CASPT2 PIPELINE: Lithium Atom (cc-pVQZ)")
print("==========================================================\n")

# 1. SETUP
mol = mshqc.Molecule()
mol.add_atom(3, 0.0, 0.0, 0.0) # Li
mol.set_multiplicity(2)
mol.set_charge(0)

basis = mshqc.BasisSet("cc-pVQZ", mol)
integrals = mshqc.IntegralEngine(mol, basis)

print(f"Basis: cc-pVQZ ({basis.n_basis_functions()} functions)")
print("Target: Standard Algorithm (No Cholesky)")

# 2. UHF
print("\n[STEP 1] Standard UHF...")
t0 = time.time()
# Hitung alpha/beta: Li (3e, double) -> n_alpha=2, n_beta=1
uhf = mshqc.UHF(mol, basis, integrals, 2, 1)
uhf_res = uhf.compute()
print(f"    E(UHF) = {uhf_res.energy_total:.8f} Ha")
print(f"    Time   = {time.time() - t0:.4f} s")

# 3. CASSCF
print("\n[STEP 2] Standard CASSCF...")
active_space = mshqc.ActiveSpace.CAS(1, 4, basis.n_basis_functions(), 3)
casscf = mshqc.CASSCF(mol, basis, integrals, active_space)
casscf.set_max_iterations(50)  # Sekarang fungsi ini bisa dipanggil

t0 = time.time()
cas_res = casscf.compute(uhf_res)
print(f"    E(CASSCF) = {cas_res.e_casscf:.8f} Ha")
print(f"    Time      = {time.time() - t0:.4f} s")

# 4. CASPT2
print("\n[STEP 3] Standard CASPT2...")
caspt2 = mshqc.CASPT2(mol, basis, integrals, cas_res)

t0 = time.time()
pt2_res = caspt2.compute()
print(f"    E(Total)  = {pt2_res.e_total:.8f} Ha")
print(f"    Time      = {time.time() - t0:.4f} s")