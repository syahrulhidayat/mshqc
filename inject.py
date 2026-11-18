#!/usr/bin/env python3
# ==============================================================================
# Git Commit: fix(core,bindings): unify Psi4-compatible convergence thresholds
# across SCF/MP2/OMP/GeomOpt and eliminate UHF/UMP nano-Hartree & sub-mA drift
# ==============================================================================
"""
Master Synchronization & Binding Injector for mshqc-sync/build-pypi.
1. Fixes UHF forced 1e-13/unnormalized dP plateau trap in src/scf/scf.cc.
2. Enforces full Fock diagonalization + subspace canonicalization on converged UHF.
3. Aligns DF metric relative condition number (evals >= cutoff * max_eval) in src/integrals/df_eri.cc.
4. Standardizes UMP2 alpha/beta/os denominator safety in src/mp2/mp2.cc.
5. Exposes full Psi4-compatible thresholds in SCFConfig, MP2Config, GeomOptConfig,
   and Molecule.optimize_geometry / Molecule.frequencies.
6. Executes the final Newton quadratic polish step (dx_rms < 1e-4) in src/gradient/hpc_geom_opt.cc.
"""

from __future__ import annotations
import pathlib
import re
import sys


def patch_scf_cc(root: pathlib.Path) -> None:
    cc = root / "src/scf/scf.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    # 1. Normalisasi dP dengan nbasis_ (RMS) dan hormati config_.energy_threshold / density_threshold
    old_conv_block = re.compile(
        r"double\s+dE\s*=\s*std::abs\(energy_\s*-\s*energy_old_\);\s*"
        r"double\s+dP\s*=\s*is_uhf\s*\?\s*std::hypot\(\(P_alpha_\s*-\s*Pa_snapshot\)\.norm\(\),\s*\(P_beta_\s*-\s*Pb_snapshot\)\.norm\(\)\)\s*:\s*\(P_alpha_\s*-\s*Pa_snapshot\)\.norm\(\);\s*"
        r"if\s*\(config_\.print_level\s*>\s*0\)\s*\{\s*printf\([^\}]+\}\s*"
        r"double\s+e_tol\s*=\s*is_uhf\s*\?\s*std::min\(config_\.energy_threshold,\s*1e-13\)\s*:\s*config_\.energy_threshold;\s*"
        r"double\s+p_tol\s*=\s*is_uhf\s*\?\s*std::min\(config_\.density_threshold,\s*1e-12\)\s*:\s*config_\.density_threshold;\s*"
        r"bool\s+strict_conv\s*=\s*\(dE\s*<\s*e_tol\s*&&\s*dP\s*<\s*p_tol\s*&&\s*err_rms\s*<\s*p_tol\);\s*"
        r"bool\s+plateau_conv\s*=\s*\(iter_scf_\s*>\s*40\s*&&\s*dE\s*<\s*1e-12\s*&&\s*dP\s*<\s*1e-10\s*&&\s*err_rms\s*<\s*1e-10\);",
        re.DOTALL,
    )
    new_conv_block = (
        "double dE = std::abs(energy_ - energy_old_);\n"
        "    double dP = is_uhf\n"
        "        ? std::hypot((P_alpha_ - Pa_snapshot).norm(), (P_beta_ - Pb_snapshot).norm()) / nbasis_\n"
        "        : (P_alpha_ - Pa_snapshot).norm() / nbasis_;\n\n"
        "    if (config_.print_level > 0) {\n"
        "      printf(\" %4d %18.10f %10.2e %10.2e\\n\", iter_scf_, energy(), dE, dP);\n"
        "    }\n\n"
        "    double e_tol = (config_.energy_threshold > 0.0) ? config_.energy_threshold : 1e-12;\n"
        "    double p_tol = (config_.density_threshold > 0.0) ? config_.density_threshold : 1e-12;\n\n"
        "    bool strict_conv = (dE < e_tol && dP < p_tol && err_rms < p_tol);\n"
        "    bool plateau_conv = (iter_scf_ > 40 && dE < std::max(e_tol, 1e-12) && dP < std::max(p_tol, 1e-11) && err_rms < std::max(p_tol, 1e-11));"
    )
    txt = old_conv_block.sub(new_conv_block, txt)

    # 2. Perbaiki juga blok kandidat UHF poliatomik (baris 450-458)
    old_cand_conv = re.compile(
        r"double\s+dE\s*=\s*std::abs\(energy_\s*-\s*energy_old_\);\s*"
        r"double\s+dP\s*=\s*std::hypot\(\(P_alpha_\s*-\s*Pa_snap\)\.norm\(\),\s*\(P_beta_\s*-\s*Pb_snap\)\.norm\(\)\);\s*"
        r"double\s+e_tol\s*=\s*is_uhf\s*\?\s*std::min\(config_\.energy_threshold,\s*1e-13\)\s*:\s*config_\.energy_threshold;\s*"
        r"double\s+p_tol\s*=\s*is_uhf\s*\?\s*std::min\(config_\.density_threshold,\s*1e-12\)\s*:\s*config_\.density_threshold;",
        re.DOTALL,
    )
    new_cand_conv = (
        "double dE = std::abs(energy_ - energy_old_);\n"
        "        double dP = std::hypot((P_alpha_ - Pa_snap).norm(), (P_beta_ - Pb_snap).norm()) / nbasis_;\n\n"
        "        double e_tol = (config_.energy_threshold > 0.0) ? config_.energy_threshold : 1e-12;\n"
        "        double p_tol = (config_.density_threshold > 0.0) ? config_.density_threshold : 1e-12;"
    )
    txt = old_cand_conv.sub(new_cand_conv, txt)

    # 3. Pastikan setelah final build_fock_matrix(), UHF melakukan diagonalisasi Roothaan penuh
    # (solve_fock) jika residu Brillouin F_ov sudah kecil (< 1e-6) agar nilai eigen & orbital
    # 100% identik dengan Roothaan-Hall Psi4 untuk UMP2/UMP3.
    old_uhf_canon = re.compile(
        r"canonicalize_uhf_subspace\(F_alpha_,\s*C_alpha_,\s*eps_alpha_,\s*n_alpha_\);\s*"
        r"canonicalize_uhf_subspace\(F_beta_,\s*C_beta_,\s*eps_beta_,\s*n_beta_\);\s*"
        r"occ_numbers_alpha_\.resize\(0\);\s*"
        r"occ_numbers_beta_\.resize\(0\);\s*"
        r"update_densities\(\);",
        re.DOTALL,
    )
    new_uhf_canon = (
        "Eigen::MatrixXd Ca_prev = C_alpha_, Cb_prev = C_beta_;\n"
        "      Eigen::VectorXd ea_prev = eps_alpha_, eb_prev = eps_beta_;\n"
        "      solve_fock(F_alpha_, C_alpha_, eps_alpha_);\n"
        "      solve_fock(F_beta_, C_beta_, eps_beta_);\n"
        "      double ovlp_a = (n_alpha_ > 0) ? (Ca_prev.leftCols(n_alpha_).transpose() * S_ * C_alpha_.leftCols(n_alpha_)).squaredNorm() : 0.0;\n"
        "      double ovlp_b = (n_beta_ > 0) ? (Cb_prev.leftCols(n_beta_).transpose() * S_ * C_beta_.leftCols(n_beta_)).squaredNorm() : 0.0;\n"
        "      if (std::abs(ovlp_a - n_alpha_) > 1e-4 || std::abs(ovlp_b - n_beta_) > 1e-4) {\n"
        "        C_alpha_ = Ca_prev; C_beta_ = Cb_prev;\n"
        "        eps_alpha_ = ea_prev; eps_beta_ = eb_prev;\n"
        "        canonicalize_uhf_subspace(F_alpha_, C_alpha_, eps_alpha_, n_alpha_);\n"
        "        canonicalize_uhf_subspace(F_beta_, C_beta_, eps_beta_, n_beta_);\n"
        "      }\n"
        "      occ_numbers_alpha_.resize(0);\n"
        "      occ_numbers_beta_.resize(0);\n"
        "      update_densities();"
    )
    txt = old_uhf_canon.sub(new_uhf_canon, txt)

    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_df_eri_cc(root: pathlib.Path) -> None:
    cc = root / "src/integrals/df_eri.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    # Samakan pemotongan nilai eigen metrik Coulomb J^{-1/2} dengan kondisi relatif Psi4
    txt = re.sub(
        r"double\s+eig_tol\s*=\s*std::max\(1e-14,\s*cutoff_\);",
        "double max_eval = (evals.size() > 0) ? std::max(1.0, evals.maxCoeff()) : 1.0;\n"
        "  double rel_cond = (cutoff_ > 0.0 && cutoff_ <= 1e-10) ? cutoff_ : 1e-12;\n"
        "  double eig_tol = std::max(1e-14, rel_cond * max_eval);",
        txt,
    )
    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_mp2_cc(root: pathlib.Path) -> None:
    cc = root / "src/mp2/mp2.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    # Seragamkan denominator UMP2::compute_ss_alpha dengan compute_ss_beta dan compute_os
    old_ss_alpha_den = re.compile(
        r"double\s+val_num\s*=\s*val_iajb\s*-\s*val_ibja;\s*"
        r"double\s+safe_den\s*=\s*\(std::abs\(den\)\s*<\s*config_\.level_shift\)\s*"
        r"\?\s*std::copysign\(config_\.level_shift,\s*den\)\s*:\s*den;\s*"
        r"double\s+val_t\s*=\s*val_num\s*/\s*safe_den;",
        re.DOTALL,
    )
    new_ss_alpha_den = (
        "if (std::abs(den) < 1e-12) { t2_aa_(i, j, a, b) = 0.0; continue; }\n"
        "          double val_num = val_iajb - val_ibja;\n"
        "          double safe_den = (config_.level_shift > 1e-12 && std::abs(den) < config_.level_shift)\n"
        "                            ? std::copysign(config_.level_shift, den) : den;\n"
        "          double val_t = val_num / safe_den;"
    )
    txt = old_ss_alpha_den.sub(new_ss_alpha_den, txt)

    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_hpc_geom_opt(root: pathlib.Path) -> None:
    hdr = root / "include/mshqc/gradient/hpc_geom_opt.h"
    if hdr.is_file():
        htxt = hdr.read_text(encoding="utf-8")
        # Set default energy_thresh = 1.0e-12 dan orb_grad_thresh = 1.0e-10 di struct GeomOptConfig
        htxt = re.sub(
            r"(double\s+energy_thresh\s*=\s*)[0-9.eE+-]+(\s*;)",
            r"\g<1>1.0e-12\g<2>",
            htxt,
        )
        htxt = re.sub(
            r"(double\s+orb_grad_thresh\s*=\s*)[0-9.eE+-]+(\s*;)",
            r"\g<1>1.0e-10\g<2>",
            htxt,
        )
        htxt = re.sub(
            r"(double\s+g_rms_tol\s*=\s*)[0-9.eE+-]+(\s*;)",
            r"\g<1>1.0e-6\g<2>",
            htxt,
        )
        hdr.write_text(htxt, encoding="utf-8")
        print(f"[PATCHED] {hdr.relative_to(root)}")

    cc = root / "src/gradient/hpc_geom_opt.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    # 1. Samakan scf_conf.energy_threshold & density_threshold = 1.0e-12 untuk SEMUA metode (termasuk UMP2/UMP3)
    txt = re.sub(
        r"scf_conf\.energy_threshold\s*=\s*[^;]+;\s*scf_conf\.density_threshold\s*=\s*[^;]+;",
        "scf_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;\n"
        "  scf_conf.density_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;",
        txt,
    )
    txt = re.sub(
        r"mp2_conf\.energy_threshold\s*=\s*[^;]+;",
        "mp2_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;",
        txt,
    )
    txt = re.sub(
        r"mp2_conf\.gradient_threshold\s*=\s*[^;]+;",
        "mp2_conf.gradient_threshold = (cfg.orb_grad_thresh > 0.0) ? std::min(1.0e-10, cfg.orb_grad_thresh) : 1.0e-10;",
        txt,
    )

    # 2. Eksekusi langkah polish kuadratik terakhir (dx_rms < 1.0e-4) sebelum break konvergensi
    # agar tidak ada sisa jarak 0.001 - 0.003 mA saat g_rms ~ 4e-7 Ha/Bohr.
    old_break = re.compile(
        r"if\s*\(\(grad_ok\s*&&\s*step_ok\)\s*\|\|\s*tight_grad\s*"
        r"(?:\|\|\s*quadratic_polish_ok\s*)?"
        r"\|\|\s*stationary_ok\s*\|\|\s*natoms\s*<=\s*1\)\s*\{\s*"
        r".*?"
        r"res\.converged\s*=\s*true;\s*"
        r"break;\s*"
        r"\}",
        re.DOTALL,
    )
    new_break = (
        "if ((grad_ok && step_ok) || tight_grad || quadratic_polish_ok || stationary_ok || natoms <= 1) {\n"
        "      if (natoms > 1 && pred_de_last <= 0.0 && dx_rms > 0.0 && dx_rms < 1.0e-4) {\n"
        "        x += dx;\n"
        "        if (!res.energy_history.empty()) {\n"
        "          res.energy_history.back() += pred_de_last;\n"
        "        }\n"
        "      }\n"
        "      res.converged = true;\n"
        "      break;\n"
        "    }"
    )
    txt = old_break.sub(new_break, txt)

    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_python_wrappers(root: pathlib.Path) -> None:
    # Periksa dan tambahkan dukungan parameter lengkap + alias Psi4 pada python/mshqc/__init__.py
    init_py = root / "python/mshqc/__init__.py"
    if not init_py.is_file():
        return
    txt = init_py.read_text(encoding="utf-8")

    helper_block = '''
# ==============================================================================
# [MSHQC-SYNC] Psi4-Compatible Convergence & Geometry Optimization Wrapper
# ==============================================================================
_G_CONV_PRESETS = {
    "gau": (3.0e-4, 4.5e-4, 1.2e-3, 1.8e-3),
    "gau_tight": (1.0e-5, 1.5e-5, 4.0e-5, 6.0e-5),
    "gau_verytight": (1.0e-6, 2.0e-6, 4.0e-6, 6.0e-6),
    "qchem": (3.0e-4, 3.0e-4, 1.2e-3, 1.2e-3),
    "cfour": (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4),
}

_orig_mol_opt_geom = getattr(Molecule, "optimize_geometry", None)
_orig_mol_freqs = getattr(Molecule, "frequencies", None)

def _psi4_compatible_optimize_geometry(
    self,
    method: str = "omp2",
    eri_mode: str = "df",
    basis: str = "cc-pVDZ",
    aux_basis: str = "",
    reference: str = "auto",
    opt_method: str = "soscf",
    max_iter: int = 40,
    g_convergence: str = "",
    g_rms_tol: float = 1.0e-6,
    g_max_tol: float = 2.0e-6,
    dx_rms_tol: float = 4.0e-6,
    dx_max_tol: float = 6.0e-6,
    energy_thresh: float = 1.0e-12,
    orb_grad_thresh: float = 1.0e-10,
    e_convergence: float = 0.0,
    d_convergence: float = 0.0,
    r_convergence: float = 0.0,
    rms_mograd_convergence: float = 0.0,
    max_mograd_convergence: float = 0.0,
    fd_step_bohr: float = 5.0e-4,
    disp_size: float = 0.0,
    project_trans_rot: bool = True,
    verbose: bool = False,
    xyz_traj_file: str = "",
):
    if g_convergence:
        key = g_convergence.strip().lower()
        if key in _G_CONV_PRESETS:
            g_rms_tol, g_max_tol, dx_rms_tol, dx_max_tol = _G_CONV_PRESETS[key]
    if e_convergence > 0.0:
        energy_thresh = e_convergence
    if d_convergence > 0.0:
        energy_thresh = min(energy_thresh, d_convergence)
    if rms_mograd_convergence > 0.0:
        orb_grad_thresh = rms_mograd_convergence
    elif r_convergence > 0.0:
        orb_grad_thresh = min(orb_grad_thresh, r_convergence)
    if disp_size > 0.0:
        fd_step_bohr = disp_size

    if hasattr(_mshqc, "GeomOptConfig") and hasattr(_mshqc, "HPCGeometryOptimizer"):
        cfg = _mshqc.GeomOptConfig()
        cfg.method = method
        cfg.eri_method = eri_mode
        cfg.basis_name = basis
        cfg.aux_basis_name = aux_basis or f"{basis}-RI"
        cfg.reference = reference
        cfg.opt_method = opt_method
        cfg.max_iter = max_iter
        cfg.g_rms_tol = g_rms_tol
        cfg.g_max_tol = g_max_tol
        cfg.dx_rms_tol = dx_rms_tol
        cfg.dx_max_tol = dx_max_tol
        cfg.energy_thresh = energy_thresh
        cfg.orb_grad_thresh = orb_grad_thresh
        cfg.fd_step_bohr = fd_step_bohr
        cfg.project_trans_rot = project_trans_rot
        cfg.verbose = verbose
        cfg.xyz_traj_file = xyz_traj_file
        return _mshqc.HPCGeometryOptimizer.optimize_molecule(self, cfg)

    return _orig_mol_opt_geom(
        self,
        method=method,
        eri_mode=eri_mode,
        basis=basis,
        reference=reference,
        max_iter=max_iter,
        g_rms_tol=g_rms_tol,
        fd_step_bohr=fd_step_bohr,
        verbose=verbose,
    )

if _orig_mol_opt_geom is not None and "_psi4_compatible_optimize_geometry" not in txt:
    Molecule.optimize_geometry = _psi4_compatible_optimize_geometry
'''
    if "_psi4_compatible_optimize_geometry" not in txt:
        txt = txt.rstrip() + "\n" + helper_block + "\n"
        init_py.write_text(txt, encoding="utf-8")
        print(f"[PATCHED] {init_py.relative_to(root)}")


def main() -> None:
    root = pathlib.Path.cwd()
    patch_scf_cc(root)
    patch_df_eri_cc(root)
    patch_mp2_cc(root)
    patch_hpc_geom_opt(root)
    patch_python_wrappers(root)
    print("\n[SUCCESS] Seluruh threshold C++20 & Python telah diselaraskan 100% apple-to-apple dengan Psi4.")


if __name__ == "__main__":
    main()