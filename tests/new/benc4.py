import mshqc

# Mode 1: High-Level One-Liner (Otomatis PointGroup + PetiteList + C++20 HPC)
res = mshqc.run(
    "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692",
    method=["RHF", "RMP2", "OMP2", "OMP3"],
    basis="cc-pVTZ",
    eri_method="cholesky",
    units="angstrom",
)

# Mode 2: Low-Level C++20 Objects (Tanpa Boilerplate Rumit)
mol = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
basis = mshqc.BasisSet("cc-pVTZ", mol)
ints = mshqc.IntegralEngine(mol, basis)

scf_conf = mshqc.SCFConfig(eri_method="cholesky", threshold=1e-9)
scf_res = mshqc.RHF(mol, basis, ints, scf_conf).compute()

mp2_conf = mshqc.MP2Config(eri_method="cholesky", threshold=1e-9)
mp2_res = mshqc.RMP2(mol, basis, ints, scf_res, mp2_conf).compute()
omp2_res = mshqc.OMP2(mol, basis, ints, scf_res, mp2_conf).compute()
omp3_res = mshqc.OMP3(mol, basis, ints, scf_res, mp2_conf).compute()