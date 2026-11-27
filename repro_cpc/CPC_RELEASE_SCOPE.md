# mshqc v2.0.0 — CPC (Computer Programs in Physics) Release Scope

## 1. Feature Freeze (Included in Manuscript #1)
- **Architecture:** Thin-core C++20 (9 core files, zero-HDF5, zero-copy TBLIS/BLAS mapping) + Python orchestration + PySCF interoperability (`run_from_pyscf`) + orbital optimization plugin hooks (`optimize_orbitals_rhf`, `optimize_orbitals_uhf`).
- **References:** RHF, UHF, ROHF.
- **Orbital-Optimized Perturbation Methods:** OMP2, SCS-OMP2, SOS-OMP2, OMP2.5, OMP3 across `EXACT`, `CHOLESKY`, and `DF` integral modes.
- **Reduced-Scaling & Regularized Extensions:** O(N^4) Zero-T2 `LT-SOS-OMP2` and regularized `kappa-OOMP2`.
- **Analytic Nuclear Gradients:** DF-SCF, DF-OMP2/SCS/SOS, and DF-kappa-OOMP2 (All-Electron & Strict Frozen-Core Z-vector, verified <= 1e-7 Ha/Bohr vs 4-point O(h^4) FD).
- **GPU Acceleration:** PyTorch CUDA FP64 backend (`torch_backend.py`) benchmarked on datacenter FP64 hardware (Modal A100/H100).

## 2. Quarantined / Deferred (Excluded from v2.0.0 CPC Manuscript)
- `experimental/oo_dh_dft/`: deferred until full exchange-correlation kernel response term `f_xc[P_corr]` passes central finite-difference verification.
- Multi-reference solvers (DMRG, CASSCF, MC-PDFT): out of scope for Paper #1; planned for ecosystem interoperability via OO-MP2 natural orbitals.

## 3. Phase 0 Exit Gates (Completed)
- [x] Zero active references to `oo_dh_dft` in public `mshqc` exports or default test suite.
- [x] `test_mshqc_hybrid.yml` 8/8 fast CPU verification passing on `ubuntu-latest` and `macos-latest`.
- [x] Heavy Bab IV/Psi4 jobs gated behind `workflow_dispatch` (`run_full_benchmarks=true`).
- [x] Single canonical Modal GPU runner (`.modal/gpu_benchmark.py` + `gpu_cuda_modal.yml`).

## 4. Phase 1 Exit Gates (Active)
- [ ] `tests/test_phase1_cpc_verification.py` green in `test_mshqc_hybrid.yml` (FD4 <= 1e-7 Ha/Bohr, rotational torque < 1e-9, Cholesky/Laplace/kappa bounds, edge cases).
- [ ] `cpc_sanitizers.yml` executed via `workflow_dispatch`: `phase1_error_bounds.json`, `01_asan_ubsan_log.txt`, and `02_tsan_openmp_log.txt` archived.
- [ ] `gpu_cuda_modal.yml` smoke (`T4`, `mode=smoke`) green before running A100 `cpc_scaling` in Phase 2.
