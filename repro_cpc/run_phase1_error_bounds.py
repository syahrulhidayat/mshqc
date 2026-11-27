#!/usr/bin/env python3
"""
repro_cpc/run_phase1_error_bounds.py
Menghasilkan tabel & JSON mentah batas error terkendali Fase 1 untuk naskah CPC:
  1. Paritas gradien nuklir analitik vs 4-point O(h^4) FD + invarian translasi & rotasi.
  2. Konvergensi ambang Cholesky tau_CD terhadap EXACT ERI (RHF, OMP2, OMP3).
  3. Akurasi kuadratur Laplace NNLS & LT-SOS-OMP2 lintas rasio celah R (termasuk H2O terenggang).
  4. Sapuan regularisasi kappa-OOMP2 pada H2O setimbang dan H2O terenggang (R_OH = 1.8 A).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

import mshqc
from mshqc import kernels as K
from mshqc import methods as M
from mshqc.methods.lt_sos_omp2 import build_laplace_quadrature


def _central_fd4(mol, method: str, basis: str, ref: str, freeze_core: bool = False, kappa: float = 1.45, h: float = 5e-4):
    nat = int(mol.n_atoms())
    x0 = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(nat)], dtype=np.float64)
    g_fd = np.zeros((nat, 3), dtype=np.float64)
    for a in range(nat):
        for d in range(3):
            acc = 0.0
            for step, w in ((-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)):
                m_d = mshqc.Molecule(int(mol.charge()), int(mol.multiplicity()))
                for i in range(nat):
                    p = x0[i].copy()
                    if i == a:
                        p[d] += step * h
                    m_d.add_atom(int(mol.atom(i).atomic_number), float(p[0]), float(p[1]), float(p[2]))
                e_val, _ = M.compute_df_gradient(
                    m_d, method=method, basis_name=basis, reference=ref,
                    threshold=1e-12, orb_grad_tol=1e-10, kappa=kappa, freeze_core=freeze_core,
                )
                acc += w * e_val
            g_fd[a, d] = acc / (12.0 * h)
    return g_fd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CPC Phase 1 error-bound tables and raw JSON.")
    parser.add_argument("--out-json", default="phase1_error_bounds.json")
    parser.add_argument("--out-md", default="phase1_error_bounds.md")
    args = parser.parse_args()

    t0 = time.perf_counter()

    # --- 1. Analytic Gradient FD4 + Invariance Table ---
    h2o_asym = mshqc.Molecule(0, 1)
    h2o_asym.add_atom(8, 0.00, 0.00, 0.22)
    h2o_asym.add_atom(1, 0.15, 1.48, -0.85)
    h2o_asym.add_atom(1, -0.10, -1.40, -0.92)

    oh_mol = mshqc.Molecule(0, 2)
    oh_mol.add_atom(8, 0.00, 0.00, 0.00)
    oh_mol.add_atom(1, 0.25, -0.15, 1.88)

    grad_rows = []
    for label, mol, method, ref, fc in (
        ("H2O (RHF, AE)", h2o_asym, "omp2", "rhf", False),
        ("H2O (RHF, FC)", h2o_asym, "omp2", "rhf", True),
        ("H2O (RHF, AE)", h2o_asym, "kappa-omp2", "rhf", False),
        ("H2O (RHF, FC)", h2o_asym, "kappa-omp2", "rhf", True),
        ("OH  (UHF, AE)", oh_mol, "omp2", "uhf", False),
        ("OH  (UHF, AE)", oh_mol, "kappa-omp2", "uhf", False),
    ):
        _, g_ana = M.compute_df_gradient(
            mol, method=method, basis_name="cc-pVDZ", reference=ref,
            threshold=1e-12, orb_grad_tol=1e-10, kappa=1.45, freeze_core=fc,
        )
        g_fd = _central_fd4(mol, method, "cc-pVDZ", ref, freeze_core=fc, kappa=1.45)
        coords = np.array([[mol.atom(i).x, mol.atom(i).y, mol.atom(i).z] for i in range(int(mol.n_atoms()))])
        grad_rows.append({
            "system": label,
            "method": f"DF-{method.upper()}",
            "max_fd4_err_ha_bohr": float(np.max(np.abs(g_ana - g_fd))),
            "trans_err_ha_bohr": float(np.max(np.abs(np.sum(g_ana, axis=0)))),
            "rot_torque_err_ha": float(np.max(np.abs(np.sum(np.cross(coords, g_ana), axis=0)))),
        })

    # --- 2. Cholesky tau_CD Error Convergence (H2O cc-pVDZ) ---
    h2o_eq = mshqc.Molecule(0, 1)
    h2o_eq.add_atom("O", 0.0, 0.0, 0.1173, units="angstrom")
    h2o_eq.add_atom("H", 0.0, 0.7572, -0.4692, units="angstrom")
    h2o_eq.add_atom("H", 0.0, -0.7572, -0.4692, units="angstrom")

    c_ex = mshqc.MSHQCCalculator(h2o_eq, basis_name="cc-pVDZ", eri_method="exact", threshold=1e-11)
    s_ex = c_ex.run_scf()
    o2_ex = c_ex.run_omp2(s_ex, grad_thresh=1e-7)
    o3_ex = c_ex.run_omp3(s_ex, grad_thresh=1e-7)

    cd_rows = []
    for tau in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        c_cd = mshqc.MSHQCCalculator(h2o_eq, basis_name="cc-pVDZ", eri_method="cholesky", threshold=tau)
        s_cd = c_cd.run_scf()
        o2_cd = c_cd.run_omp2(s_cd, grad_thresh=1e-7)
        o3_cd = c_cd.run_omp3(s_cd, grad_thresh=1e-7)
        cd_rows.append({
            "tau_cd": tau,
            "de_rhf_ha": abs(s_cd.energy_total - s_ex.energy_total),
            "de_omp2_ha": abs(o2_cd.energy_total - o2_ex.energy_total),
            "de_omp3_ha": abs(o3_cd.energy_total - o3_ex.energy_total),
        })

    # --- 3. Laplace Quadrature & kappa Sweep (Equilibrium vs Stretched H2O R=1.8 A) ---
    h2o_str = mshqc.Molecule(0, 1)
    theta = np.deg2rad(104.5 / 2.0)
    r_str = 1.80
    h2o_str.add_atom("O", 0.0, 0.0, 0.0, units="angstrom")
    h2o_str.add_atom("H", 0.0, r_str * np.sin(theta), r_str * np.cos(theta), units="angstrom")
    h2o_str.add_atom("H", 0.0, -r_str * np.sin(theta), r_str * np.cos(theta), units="angstrom")

    laplace_rows = []
    kappa_rows = []
    for geom_label, mol in (("H2O_eq (0.96 A)", h2o_eq), ("H2O_stretched (1.80 A)", h2o_str)):
        basis = mshqc.BasisSet("cc-pVDZ", mol)
        aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
        ints = mshqc.IntegralEngine(mol, basis)
        H = ints.compute_kinetic() + ints.compute_nuclear()
        S = ints.compute_overlap()
        v_nn = float(mol.nuclear_repulsion_energy())
        B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10, False, False)
        scf_r = M.run_df_rhf(H, S, v_nn, B_ao, 5, energy_tol=1e-11, density_tol=1e-11)

        eps = scf_r.orbital_energies_alpha
        d_ov = (eps[5:][None, :] - eps[:5][:, None]).reshape(-1)
        d_min, d_max = 2.0 * float(np.min(d_ov)), 2.0 * float(np.max(d_ov))
        w, t = build_laplace_quadrature(d_min, d_max, n_grid=220)
        sos_ref = M.run_df_omp_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, method_name="sos-omp2", c_os=1.3, c_ss=0.0, grad_tol=1e-7)
        lt_res = M.run_df_lt_sos_omp2_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, c_os=1.3, grad_tol=1e-7)
        laplace_rows.append({
            "system": geom_label,
            "gap_ratio_R": round(d_max / d_min, 2),
            "n_q_active": int(len(w)),
            "e_sos_omp2": float(sos_ref.energy_total),
            "e_lt_sos_omp2": float(lt_res.energy_total),
            "abs_de_ha": abs(lt_res.energy_total - sos_ref.energy_total),
        })

        omp2_r = M.run_df_omp_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, method_name="omp2", grad_tol=1e-7)
        for kv in (0.5, 1.0, 1.45, 2.0, 4.0, 10.0, 50.0):
            kr = M.run_df_kappa_omp2_rhf(H, v_nn, B_ao, scf_r.C_alpha, 5, kappa=kv, grad_tol=1e-7)
            kappa_rows.append({
                "system": geom_label,
                "kappa": kv,
                "energy_ha": float(kr.energy_total),
                "iterations": int(kr.iterations),
                "converged": bool(kr.converged),
                "delta_vs_omp2_mha": (kr.energy_total - omp2_r.energy_total) * 1e3,
            })

    md = [
        f"### mshqc v{mshqc.__version__} Phase 1 CPC Verification & Error Bounds",
        "",
        "#### 1. Analytic Nuclear Gradients vs 4-Point O(h^4) Central FD (`cc-pVDZ`)",
        "| System | Method | max |ana - FD4| (Ha/Bohr) | Translational Sum | Rotational Torque (Ha) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in grad_rows:
        md.append(
            f"| {r['system']} | {r['method']} | {r['max_fd4_err_ha_bohr']:.2e} | "
            f"{r['trans_err_ha_bohr']:.2e} | {r['rot_torque_err_ha']:.2e} |"
        )

    md += [
        "",
        "#### 2. Cholesky Threshold Convergence (`CHOLESKY` vs `EXACT`, H2O `cc-pVDZ`)",
        "| tau_CD | |dE_RHF| (Ha) | |dE_OMP2| (Ha) | |dE_OMP3| (Ha) |",
        "|---:|---:|---:|---:|",
    ]
    for r in cd_rows:
        md.append(f"| {r['tau_cd']:.0e} | {r['de_rhf_ha']:.2e} | {r['de_omp2_ha']:.2e} | {r['de_omp3_ha']:.2e} |")

    md += [
        "",
        "#### 3. O(N^4) Laplace-Transformed SOS-OMP2 Parity Across Gap Ratios",
        "| System | Gap Ratio R | Active n_q | |E_LT-SOS - E_SOS| (Ha) |",
        "|---|---:|---:|---:|",
    ]
    for r in laplace_rows:
        md.append(f"| {r['system']} | {r['gap_ratio_R']:.2f} | {r['n_q_active']} | {r['abs_de_ha']:.2e} |")

    md += [
        "",
        "#### 4. Regularized kappa-OOMP2 Parameter Sweep (Equilibrium vs Stretched H2O)",
        "| System | kappa | E_total (Ha) | Iter | E(kappa) - E(OMP2) (mHa) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in kappa_rows:
        md.append(
            f"| {r['system']} | {r['kappa']:.2f} | {r['energy_ha']:.8f} | "
            f"{r['iterations']} | {r['delta_vs_omp2_mha']:+.4f} |"
        )

    md_text = "\n".join(md) + "\n"
    payload = {
        "mshqc_version": mshqc.__version__,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "gradients_fd4": grad_rows,
        "cholesky_convergence": cd_rows,
        "laplace_parity": laplace_rows,
        "kappa_sweep": kappa_rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(md_text, encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()
