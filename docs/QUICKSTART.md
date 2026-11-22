# Quickstart Guide: Using & Extending `mshqc`

This guide covers all three tiers of `mshqc`:
1. **High-Level Pipeline API** (`mshqc.run`, `MSHQCCalculator`, `recommend_hpc_settings`, geometry optimization)
2. **Modular Python Methods & Interoperability** (`mshqc.methods`: `run_from_arrays`, `run_from_pyscf`, `optimize_orbitals_rhf`, 4D `einsum` helpers)
3. **Coarse-Grained C++20 HPC Kernels** (`mshqc.kernels`)

---

## 1. Thread Affinity & Cross-Platform Environment

On `import mshqc`, the built-in auto-tuner detects physical CPU cores and sets `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` to avoid nested OpenMP/BLAS oversubscription. On **macOS and Windows**, `TBLIS_NUM_THREADS=1` is automatically applied so TBLIS pthreads do not contend with OpenMP threads.

For explicit control on Linux HPC nodes:
```bash
export OMP_NUM_THREADS=4
export TBLIS_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
```

---

## 2. High-Level Pipeline API (`mshqc.run`)

`mshqc.run()` supports both closed-shell (`RHF`) and open-shell (`UHF`/`ROHF`) references across `"exact"`, `"cholesky"`, and `"df"` integral modes. In `"df"` and `"cholesky"` modes, the 3-index AO factor (`SCFResult.L_mat`) is computed once and reused automatically by `SCS-OMP2`, `SOS-OMP2`, `OMP2.5`, `OMP2`, and `OMP3`:

```python
import mshqc

oh_radical = "O 0.0 0.0 0.0\nH 0.0 0.0 0.9697"
results = mshqc.run(
    oh_radical,
    method=["UHF", "OMP2", "SCS-OMP2", "OMP2.5", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",
    multiplicity=2,
    units="angstrom",
    threshold=1e-9,
    grad_thresh=1e-5,
)

for name, data in results.items():
    print(f"{name:<9}: {data['energy']:.10f} Ha ({data['time']:.3f} s)")
```

### Geometry Optimization, Frequencies & HPC Auto-Profiler

```python
import mshqc

mol = mshqc.build_molecule("O 0 0 0.12\nH 0 0.78 -0.46\nH 0 -0.78 -0.46", units="angstrom")
opt = mol.optimize_geometry(method="omp2", eri_mode="df", basis="cc-pVDZ", g_convergence="gau_tight")
vib = mol.frequencies(method="omp2", eri_mode="df", basis="cc-pVDZ")
print("Optimized Energy:", opt.final_energy, "| Frequencies (cm^-1):", vib.frequencies_cm1)

rec = mshqc.calculators.recommend_hpc_settings(mol, basis_name="cc-pVTZ", method="omp3")
print(rec)
```

---

## 3. Modular Python Methods & 4D `np.einsum` Helpers (`mshqc.methods`)

```python
import numpy as np
import mshqc
from mshqc import kernels as K, methods as M

mol = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
basis = mshqc.BasisSet("cc-pVDZ", mol)
ints = mshqc.IntegralEngine(mol, basis)
scf_res = mshqc.RHF(mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)).compute()

nbf = basis.n_basis_functions()
no, nv = scf_res.n_occ_alpha, nbf - scf_res.n_occ_alpha
B_ov = K.transform_3c_mo(scf_res.L_mat, scf_res.C_alpha[:, :no], scf_res.C_alpha[:, no:], nbf)
e_os, e_ss, T2_mat = K.eval_rmp2_df(B_ov, scf_res.orbital_energies_alpha[:no], scf_res.orbital_energies_alpha[no:])

T2_4d = M.t2_to_4d(T2_mat, no, no, nv, nv)   # shape: (no, no, nv, nv)
B_ov_3d = M.b3c_to_3d(B_ov, no, nv)          # shape: (n_aux, no, nv)
```

### Interoperability with Raw NumPy Arrays or PySCF

```python
import mshqc.methods as M

res = M.run_from_arrays(
    H_core, S_ovlp, v_nn, B_ao, C_alpha, n_occ_alpha,
    C_beta=C_beta, n_occ_beta=n_occ_beta, method="omp2.5",
)

# Directly from a PySCF density-fitted SCF object:
# res = M.run_from_pyscf(mf, method="omp3")
```

### Custom Orbital-Optimized Method Hook (`optimize_orbitals_rhf`)

```python
import mshqc.kernels as K
import mshqc.methods as M

def my_custom_corr(B_oo, B_ov, B_vv, eps_o, eps_v):
    e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
    g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, no, nv, c_os=1.3, c_ss=0.0)
    Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, no, nv, c_os=1.3, c_ss=0.0)
    return 1.3 * e_os, g_oo, g_vv, Z_vo

res = M.optimize_orbitals_rhf(H_core, v_nn, B_ao, C0, no, corr_fn=my_custom_corr, method_name="custom-sos")
```
