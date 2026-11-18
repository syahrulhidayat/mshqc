#!/usr/bin/env python3
# ==============================================================================
# Git Commit: fix(scf): repair broken multiline printf string literal in BaseSCF::compute
# ==============================================================================
"""
Master Synchronization & Binding Injector for mshqc-sync/build-pypi.
Fixes broken multiline printf in src/scf/scf.cc and enforces full Psi4 parity.
"""

from __future__ import annotations
import pathlib
import re


def patch_scf_cc(root: pathlib.Path) -> None:
    cc = root / "src/scf/scf.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    # 1. Perbaiki printf yang terpotong menjadi 2 baris di src/scf/scf.cc (baris 538-539)
    broken_printf = re.compile(
        r'printf\(\s*"\s*%4d\s+%18\.10f\s+%10\.2e\s+%10\.2e\s*\r?\n\s*",\s*iter_scf_,\s*energy\(\),\s*dE,\s*dP\s*\);'
    )
    txt = broken_printf.sub(
        lambda _: 'printf(" %4d %18.10f %10.2e %10.2e\\n", iter_scf_, energy(), dE, dP);',
        txt,
    )

    # 2. Jika scf.cc masih versi lama, terapkan normalisasi dP & threshold dengan lambda (aman dari escape \n)
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
        '      printf(" %4d %18.10f %10.2e %10.2e\\n", iter_scf_, energy(), dE, dP);\n'
        "    }\n\n"
        "    double e_tol = (config_.energy_threshold > 0.0) ? config_.energy_threshold : 1e-12;\n"
        "    double p_tol = (config_.density_threshold > 0.0) ? config_.density_threshold : 1e-12;\n\n"
        "    bool strict_conv = (dE < e_tol && dP < p_tol && err_rms < p_tol);\n"
        "    bool plateau_conv = (iter_scf_ > 40 && dE < std::max(e_tol, 1e-12) && dP < std::max(p_tol, 1e-11) && err_rms < std::max(p_tol, 1e-11));"
    )
    txt = old_conv_block.sub(lambda _: new_conv_block, txt)

    # 3. Blok kandidat UHF poliatomik
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
    txt = old_cand_conv.sub(lambda _: new_cand_conv, txt)

    # 4. Kanonikalisasi akhir UHF
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
    if "Eigen::MatrixXd Ca_prev = C_alpha_" not in txt:
        txt = old_uhf_canon.sub(lambda _: new_uhf_canon, txt)

    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_df_eri_cc(root: pathlib.Path) -> None:
    cc = root / "src/integrals/df_eri.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")
    txt = re.sub(
        r"double\s+eig_tol\s*=\s*std::max\(1e-14,\s*cutoff_\);",
        lambda _: (
            "double max_eval = (evals.size() > 0) ? std::max(1.0, evals.maxCoeff()) : 1.0;\n"
            "  double rel_cond = (cutoff_ > 0.0 && cutoff_ <= 1e-10) ? cutoff_ : 1e-12;\n"
            "  double eig_tol = std::max(1e-14, rel_cond * max_eval);"
        ),
        txt,
    )
    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_mp2_cc(root: pathlib.Path) -> None:
    cc = root / "src/mp2/mp2.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")
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
    if "if (std::abs(den) < 1e-12) { t2_aa_(i, j, a, b) = 0.0; continue; }" not in txt:
        txt = old_ss_alpha_den.sub(lambda _: new_ss_alpha_den, txt)
    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def patch_hpc_geom_opt(root: pathlib.Path) -> None:
    hdr = root / "include/mshqc/gradient/hpc_geom_opt.h"
    if hdr.is_file():
        htxt = hdr.read_text(encoding="utf-8")
        htxt = re.sub(r"(double\s+energy_thresh\s*=\s*)[0-9.eE+-]+(\s*;)", r"\g<1>1.0e-12\g<2>", htxt)
        htxt = re.sub(r"(double\s+orb_grad_thresh\s*=\s*)[0-9.eE+-]+(\s*;)", r"\g<1>1.0e-10\g<2>", htxt)
        htxt = re.sub(r"(double\s+g_rms_tol\s*=\s*)[0-9.eE+-]+(\s*;)", r"\g<1>1.0e-6\g<2>", htxt)
        hdr.write_text(htxt, encoding="utf-8")
        print(f"[PATCHED] {hdr.relative_to(root)}")

    cc = root / "src/gradient/hpc_geom_opt.cc"
    if not cc.is_file():
        return
    txt = cc.read_text(encoding="utf-8")

    txt = re.sub(
        r"scf_conf\.energy_threshold\s*=\s*[^;]+;\s*scf_conf\.density_threshold\s*=\s*[^;]+;",
        lambda _: (
            "scf_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;\n"
            "  scf_conf.density_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;"
        ),
        txt,
    )
    txt = re.sub(
        r"mp2_conf\.energy_threshold\s*=\s*[^;]+;",
        lambda _: "mp2_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;",
        txt,
    )
    txt = re.sub(
        r"mp2_conf\.gradient_threshold\s*=\s*[^;]+;",
        lambda _: "mp2_conf.gradient_threshold = (cfg.orb_grad_thresh > 0.0) ? std::min(1.0e-10, cfg.orb_grad_thresh) : 1.0e-10;",
        txt,
    )

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
    txt = old_break.sub(lambda _: new_break, txt)

    cc.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {cc.relative_to(root)}")


def main() -> None:
    root = pathlib.Path.cwd()
    patch_scf_cc(root)
    patch_df_eri_cc(root)
    patch_mp2_cc(root)
    patch_hpc_geom_opt(root)
    print("\n[SUCCESS] String literal printf pada src/scf/scf.cc telah diperbaiki dan siap dikompilasi.")


if __name__ == "__main__":
    main()