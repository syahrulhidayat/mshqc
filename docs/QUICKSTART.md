# Quickstart Guide: Using & Extending `mshqc` v2.0

This guide covers all three tiers of `mshqc` v2.0 (Pure Thin-Core, zero-HDF5):
1. **High-Level Pipeline API** (`mshqc.run`, `MSHQCCalculator`, `MSHQCSession`, `Molecule.optimize_geometry` / `frequencies` / `scan_pes`)
2. **Modular Python Methods** (`mshqc.methods`: `run_df_rhf/uhf/rohf`, `run_df_kappa_omp2_*`, `run_df_lt_sos_omp2_*`, `compute_natural_orbitals`, `write_molden`, `.npz` checkpoints, `run_from_pyscf`)
3. **Irreducible C++20 HPC Kernels** (`mshqc.kernels`)

---

## 1. Thread Affinity & Cross-Platform Environment

On `import mshqc`, the auto-tuner detects physical cores and locks `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` to avoid nested OpenMP/BLAS oversubscription. On **macOS and Windows**, `TBLIS_NUM_THREADS=1` is set automatically.

```bash
export OMP_NUM_THREADS=4
export TBLIS_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

## 2. Open-Shell Radicals, `kappa-OOMP2`, and $\mathcal{O}(N^4)$ `LT-SOS-OMP2`

```python
import mshqc

oh_radical = "O 0.0 0.0 0.0\nH 0.0 0.0 0.9697"
results = mshqc.run(
    oh_radical,
    method=["UHF", "OMP2", "SCS-OMP2", "LT-SOS-OMP2", "kappa-OMP2", "OMP2.5", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",
    multiplicity=2,
    units="angstrom",
    threshold=1e-9,
    grad_thresh=1e-5,
)

for name, data in results.items():
    print(f"{name:<12}: {data['energy']:.10f} Ha ({data['time']:.3f} s)")
```

---

## 3. Geometry Optimization, Frequencies & 1D PES Scan

```python
import mshqc

mol = mshqc.build_molecule("O 0 0 0.12\nH 0 0.78 -0.46\nH 0 -0.78 -0.46", units="angstrom")
opt = mol.optimize_geometry(method="omp2", eri_mode="df", basis="cc-pVDZ", g_convergence="gau_tight")
vib = mol.frequencies(method="omp2", eri_mode="df", basis="cc-pVDZ")
print("Optimized Energy:", opt.final_energy, "| Frequencies (cm^-1):", vib.frequencies_cm1)

h2 = mshqc.build_molecule("H 0 0 0\nH 0 0 0.74", units="angstrom")
pes = h2.scan_pes(0, 1, [0.65, 0.74, 0.90, 1.20], method="kappa-omp2", basis="cc-pVDZ")
for pt in pes:
    print(f"r = {pt.r_angstrom:.2f} A -> E = {pt.energy_ha:.8f} Ha")
```

---

## 4. Direct Kernel Access, 4D `np.einsum` Helpers & Custom Plugins

```python
import mshqc
from mshqc import kernels as K, methods as M

mol = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
basis = mshqc.BasisSet("cc-pVDZ", mol)
aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
ints = mshqc.IntegralEngine(mol, basis)

H = ints.compute_kinetic() + ints.compute_nuclear()
S = ints.compute_overlap()
v_nn = mol.nuclear_repulsion_energy()
B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10)
nbf = basis.n_basis_functions()
no = mol.n_electrons() // 2
nv = nbf - no

scf_res = M.run_df_rhf(H, S, v_nn, B_ao, no, P_init=M.build_sad_guess(mol, basis))
B_ov = K.transform_3c_mo(B_ao, scf_res.C_alpha[:, :no], scf_res.C_alpha[:, no:], nbf)
e_os, e_ss, T2_mat = K.eval_rmp2_df(B_ov, scf_res.orbital_energies_alpha[:no], scf_res.orbital_energies_alpha[no:])

T2_4d = M.t2_to_4d(T2_mat, no, no, nv, nv)  # shape: (no, no, nv, nv)
B_ov_3d = M.b3c_to_3d(B_ov, no, nv)         # shape: (n_aux, no, nv)

def custom_sos_corr(B_oo, B_ov, B_vv, eps_o, eps_v):
    e_os, _, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
    g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, no, nv, c_os=1.3, c_ss=0.0)
    Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, no, nv, c_os=1.3, c_ss=0.0)
    return 1.3 * e_os, g_oo, g_vv, Z_vo

res = M.optimize_orbitals_rhf(H, v_nn, B_ao, scf_res.C_alpha, no, corr_fn=custom_sos_corr, method_name="custom-sos")
print("Custom SOS-OMP2:", res.energy_total)
```
