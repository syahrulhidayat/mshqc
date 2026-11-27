# mshqc v2.0.0 — CPC (Computer Programs in Physics) Release Scope

## 1. Feature Freeze (Included in Manuscript #1)
- **Architecture:** Thin-core C++20 (zero-HDF5, zero-copy TBLIS/BLAS mapping) + Python orchestration + PySCF interoperability (`run_from_pyscf`) + orbital optimization plugin hooks (`optimize_orbitals_rhf`, `optimize_orbitals_uhf`).
- **References:** RHF, UHF, ROHF.
- **Orbital-Optimized Perturbation Methods:** OMP2, SCS-OMP2, SOS-OMP2, OMP2.5, OMP3 across `EXACT`, `CHOLESKY`, and `DF` integral modes.
- **Reduced-Scaling & Regularized Extensions:** O(N^4) Zero-T2 `LT-SOS-OMP2` and regularized `kappa-OOMP2`.
- **Analytic Nuclear Gradients:** DF-OMP2 and DF-kappa-OOMP2 (target finite-difference parity <= 1e-7 Ha/Bohr).
- **GPU Acceleration:** CUDA FP64 backend benchmarked on datacenter FP64 hardware (Modal A100/H100).

## 2. Quarantined / Deferred (Excluded from v2.0.0 CPC Manuscript)
- `oo_dh_dft.py` (moved to `experimental/oo_dh_dft/`): deferred until full exchange-correlation kernel response term `f_xc[P_corr]` is implemented and verified against central finite differences.
- Multi-reference methods (DMRG, CASSCF, MC-PDFT): out of scope for Paper #1; planned for ecosystem interoperability via OO-MP2 natural orbitals.

## 3. Phase 0 -> Phase 1 Exit Gates
- [ ] Zero references to `oo_dh_dft` in public `mshqc` exports or default test suite.
- [ ] `pytest tests/test_phase0_release_hygiene.py` passes cleanly.
- [ ] Separate CI jobs passing: CPU unit/parity, C++ ASan+UBSan, C++ TSan (`OMP_NUM_THREADS=2`).
- [ ] Modal FP64 GPU smoke benchmark executed via `workflow_dispatch` with JSON artifact archived.
