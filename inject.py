#!/usr/bin/env python3
"""
Menuntaskan 3 patch tersisa menggunakan pencocokan spasi fleksibel:
1. src/mp/omp_gradient.cc  -> MO-space Hessian-vector product di SOSCF & CPHF
2. src/mp/omp_fock.cc      -> Akumulasi noalias & simetrisasi non-idempoten
3. src/gradient/hpc_geom_opt.cc -> Lagrangian Pulay eksak (blok cc, oo, co) DF-ROHF
"""

from pathlib import Path
import re
import sys


def patch_omp_gradient(root: Path) -> bool:
    path = root / "src/mp/omp_gradient.cc"
    if not path.exists():
        print(f"[ERROR] File tidak ditemukan: {path}")
        return False

    text = path.read_text(encoding="utf-8")
    marker = "// [INJECTED_FINAL_OPT_MO_SOSCF]"
    if marker in text:
        print(f"[SKIP] {path} sudah terpatch ({marker})")
        return True

    pat_soscf = re.compile(
        r"Eigen::MatrixXd\s+P1_a\s*=\s*Eigen::MatrixXd::Zero\(\s*nbf_\s*,\s*nbf_\s*\)\s*;.*?"
        r"Hp\(\s*idx_h\+\+\s*\)\s*\+=\s*2\.0\s*\*\s*H_kappa_b\(\s*i\s*,\s*a\s*\)\s*;\s*\}\s*\}\s*\}",
        re.DOTALL,
    )

    new_soscf = f"""{marker}
        const int n_aux = static_cast<int>(B_ia_P_alpha_.cols());
        const bool can_use_mo_df = (config_.eri_method != "exact" && n_aux > 0 &&
                                    B_oo_P_alpha_.cols() == n_aux && B_vv_P_alpha_.cols() == n_aux);
        if (can_use_mo_df) {{
            Eigen::VectorXd w_aux = Eigen::VectorXd::Zero(n_aux);
            if (dim_a > 0) {{
                Eigen::MatrixXd ka_T = kappa_a.transpose();
                Eigen::Map<const Eigen::VectorXd> ka_vec(ka_T.data(), dim_a);
                w_aux.noalias() += (is_restricted ? 4.0 : 2.0) * (B_ia_P_alpha_.transpose() * ka_vec);
            }}
            if (!is_restricted && dim_b > 0 && B_ia_P_beta_.cols() == n_aux) {{
                Eigen::MatrixXd kb_T = kappa_b.transpose();
                Eigen::Map<const Eigen::VectorXd> kb_vec(kb_T.data(), dim_b);
                w_aux.noalias() += 2.0 * (B_ia_P_beta_.transpose() * kb_vec);
            }}

            Eigen::MatrixXd H_kappa_a = Eigen::MatrixXd::Zero(na_, va_);
            Eigen::MatrixXd H_kappa_b = (!is_restricted && dim_b > 0) ? Eigen::MatrixXd::Zero(nb_, vb_) : Eigen::MatrixXd();

            if (dim_a > 0) {{
                Eigen::VectorXd j_vo_a = B_ia_P_alpha_ * w_aux;
                Eigen::Map<const Eigen::MatrixXd> J_vo_a_mat(j_vo_a.data(), va_, na_);
                H_kappa_a.noalias() = J_vo_a_mat.transpose();
            }}
            if (!is_restricted && dim_b > 0 && B_ia_P_beta_.cols() == n_aux) {{
                Eigen::VectorXd j_vo_b = B_ia_P_beta_ * w_aux;
                Eigen::Map<const Eigen::MatrixXd> J_vo_b_mat(j_vo_b.data(), vb_, nb_);
                H_kappa_b.noalias() = J_vo_b_mat.transpose();
            }}

            const int max_t = omp_get_max_threads();
            std::vector<Eigen::MatrixXd> Ka_tls(max_t, Eigen::MatrixXd::Zero(na_, va_));
            std::vector<Eigen::MatrixXd> Kb_tls((!is_restricted && dim_b > 0) ? max_t : 0, Eigen::MatrixXd::Zero(nb_, vb_));

            #pragma omp parallel
            {{
                const int tid = omp_get_thread_num();
                Eigen::MatrixXd T_oo_a(na_, va_), T_vo_a(va_, na_);
                Eigen::MatrixXd T_oo_b, T_vo_b;
                if (!is_restricted && dim_b > 0) {{
                    T_oo_b.resize(nb_, vb_);
                    T_vo_b.resize(vb_, nb_);
                }}

                #pragma omp for schedule(static) nowait
                for (int P = 0; P < n_aux; ++P) {{
                    if (dim_a > 0) {{
                        Eigen::Map<const Eigen::MatrixXd> Boo_a(B_oo_P_alpha_.col(P).data(), na_, na_);
                        Eigen::Map<const Eigen::MatrixXd> Bvo_a(B_ia_P_alpha_.col(P).data(), va_, na_);
                        Eigen::Map<const Eigen::MatrixXd> Bvv_a(B_vv_P_alpha_.col(P).data(), va_, va_);
                        T_oo_a.noalias() = Boo_a * kappa_a;
                        Ka_tls[tid].noalias() += T_oo_a * Bvv_a;
                        T_vo_a.noalias() = Bvo_a * kappa_a;
                        Ka_tls[tid].noalias() += (T_vo_a * Bvo_a).transpose();
                    }}
                    if (!is_restricted && dim_b > 0 && B_oo_P_beta_.cols() == n_aux) {{
                        Eigen::Map<const Eigen::MatrixXd> Boo_b(B_oo_P_beta_.col(P).data(), nb_, nb_);
                        Eigen::Map<const Eigen::MatrixXd> Bvo_b(B_ia_P_beta_.col(P).data(), vb_, nb_);
                        Eigen::Map<const Eigen::MatrixXd> Bvv_b(B_vv_P_beta_.col(P).data(), vb_, vb_);
                        T_oo_b.noalias() = Boo_b * kappa_b;
                        Kb_tls[tid].noalias() += T_oo_b * Bvv_b;
                        T_vo_b.noalias() = Bvo_b * kappa_b;
                        Kb_tls[tid].noalias() += (T_vo_b * Bvo_b).transpose();
                    }}
                }}
            }}

            for (int t = 0; t < max_t; ++t) {{
                if (dim_a > 0) H_kappa_a.noalias() -= Ka_tls[t];
                if (!is_restricted && dim_b > 0) H_kappa_b.noalias() -= Kb_tls[t];
            }}

            if (dim_a > 0) {{
                int idx_h = 0;
                for (int i = 0; i < na_; ++i) {{
                    for (int a = 0; a < va_; ++a) {{
                        Hp(idx_h++) += spin_factor * H_kappa_a(i, a);
                    }}
                }}
            }}
            if (!is_restricted && dim_b > 0) {{
                int idx_h = dim_a;
                for (int i = 0; i < nb_; ++i) {{
                    for (int a = 0; a < vb_; ++a) {{
                        Hp(idx_h++) += 2.0 * H_kappa_b(i, a);
                    }}
                }}
            }}
        }} else {{
            Eigen::MatrixXd P1_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
            if (dim_a > 0) {{
                P1_a = scf_.C_alpha.leftCols(na_) * kappa_a * scf_.C_alpha.rightCols(va_).transpose();
                P1_a = (P1_a + P1_a.transpose()).eval();
            }}
            Eigen::MatrixXd P1_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
            if (!is_restricted && dim_b > 0) {{
                P1_b = scf_.C_beta.leftCols(nb_) * kappa_b * scf_.C_beta.rightCols(vb_).transpose();
                P1_b = (P1_b + P1_b.transpose()).eval();
            }} else if (is_restricted) {{
                P1_b = P1_a;
            }}
            Eigen::MatrixXd F1_a, F1_b;
            build_fock_fast(P1_a, P1_b, F1_a, F1_b);
            F1_a -= H_core_;
            if (!is_restricted || nb_ > 0) F1_b -= H_core_;
            if (dim_a > 0) {{
                Eigen::MatrixXd H_kappa_a = scf_.C_alpha.leftCols(na_).transpose() * F1_a * scf_.C_alpha.rightCols(va_);
                int idx_h = 0;
                for (int i = 0; i < na_; ++i) {{
                    for (int a = 0; a < va_; ++a) {{
                        Hp(idx_h++) += spin_factor * H_kappa_a(i, a);
                    }}
                }}
            }}
            if (!is_restricted && dim_b > 0) {{
                Eigen::MatrixXd H_kappa_b = scf_.C_beta.leftCols(nb_).transpose() * F1_b * scf_.C_beta.rightCols(vb_);
                int idx_h = dim_a;
                for (int i = 0; i < nb_; ++i) {{
                    for (int a = 0; a < vb_; ++a) {{
                        Hp(idx_h++) += 2.0 * H_kappa_b(i, a);
                    }}
                }}
            }}
        }}"""

    pat_cphf = re.compile(
        r"Eigen::MatrixXd\s+P1\s*=\s*scf_\.C_alpha\.leftCols\(\s*na\s*\)\s*\*\s*P\.transpose\(\)\s*\*\s*scf_\.C_alpha\.rightCols\(\s*va\s*\)\.transpose\(\)\s*;.*?"
        r"AP\s*\+=\s*2\.0\s*\*\s*scf_\.C_alpha\.rightCols\(\s*va\s*\)\.transpose\(\)\s*\*\s*F1_a\s*\*\s*scf_\.C_alpha\.leftCols\(\s*na\s*\)\s*;",
        re.DOTALL,
    )

    new_cphf = """const int n_aux_cphf = static_cast<int>(B_ia_P_alpha_.cols());
        if (config_.eri_method != "exact" && n_aux_cphf > 0 && na == na_ && va == va_ &&
            B_oo_P_alpha_.cols() == n_aux_cphf && B_vv_P_alpha_.cols() == n_aux_cphf) {
            Eigen::Map<const Eigen::VectorXd> p_vec_flat(P.data(), va * na);
            Eigen::VectorXd w_cphf = 4.0 * (B_ia_P_alpha_.transpose() * p_vec_flat);
            Eigen::VectorXd j_vo = B_ia_P_alpha_ * w_cphf;
            Eigen::Map<const Eigen::MatrixXd> J_vo_mat(j_vo.data(), va, na);
            AP.noalias() += 2.0 * J_vo_mat;

            Eigen::MatrixXd kappa_ov = P.transpose();
            const int max_t = omp_get_max_threads();
            std::vector<Eigen::MatrixXd> K_tls(max_t, Eigen::MatrixXd::Zero(va, na));
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                Eigen::MatrixXd T_oo(na, va), T_vo(va, na);
                #pragma omp for schedule(static) nowait
                for (int K = 0; K < n_aux_cphf; ++K) {
                    Eigen::Map<const Eigen::MatrixXd> Boo(B_oo_P_alpha_.col(K).data(), na, na);
                    Eigen::Map<const Eigen::MatrixXd> Bvo(B_ia_P_alpha_.col(K).data(), va, na);
                    Eigen::Map<const Eigen::MatrixXd> Bvv(B_vv_P_alpha_.col(K).data(), va, va);
                    T_oo.noalias() = Boo * kappa_ov;
                    K_tls[tid].noalias() += (T_oo * Bvv).transpose();
                    T_vo.noalias() = Bvo * kappa_ov;
                    K_tls[tid].noalias() += T_vo * Bvo;
                }
            }
            for (int t = 0; t < max_t; ++t) {
                AP.noalias() -= 2.0 * K_tls[t];
            }
        } else {
            Eigen::MatrixXd P1 = scf_.C_alpha.leftCols(na) * P.transpose() * scf_.C_alpha.rightCols(va).transpose();
            P1 = (P1 + P1.transpose()).eval();
            Eigen::MatrixXd F1_a, F1_b;
            build_fock_fast(P1, P1, F1_a, F1_b);
            F1_a -= H_core_;
            AP += 2.0 * scf_.C_alpha.rightCols(va).transpose() * F1_a * scf_.C_alpha.leftCols(na);
        }"""

    text, n1 = pat_soscf.subn(new_soscf, text, count=1)
    text, n2 = pat_cphf.subn(new_cphf, text, count=1)
    if n1 != 1 or n2 != 1:
        print(f"[ERROR] Gagal mencocokkan blok di {path} (soscf={n1}, cphf={n2})")
        return False

    path.write_text(text, encoding="utf-8")
    print(f"[OK]   MO-space SOSCF & CPHF berhasil diinjeksi ke {path}")
    return True


def patch_omp_fock(root: Path) -> bool:
    path = root / "src/mp/omp_fock.cc"
    if not path.exists():
        print(f"[ERROR] File tidak ditemukan: {path}")
        return False

    text = path.read_text(encoding="utf-8")
    marker = "// [INJECTED_FINAL_OPT_OMP_FOCK_SYMM]"
    if marker in text:
        print(f"[SKIP] {path} sudah terpatch ({marker})")
        return True

    pat = re.compile(
        r"Eigen::MatrixXd\s+Ka_mat\s*=\s*Eigen::MatrixXd::Zero\(\s*nbf_\s*,\s*nbf_\s*\)\s*;\s*"
        r"Eigen::MatrixXd\s+Kb_mat\s*=\s*Eigen::MatrixXd::Zero\(\s*nbf_\s*,\s*nbf_\s*\)\s*;\s*"
        r"for\s*\(\s*int\s+t\s*=\s*0\s*;\s*t\s*<\s*max_threads\s*;\s*\+\+t\s*\)\s*\{\s*"
        r"Ka_mat\s*\+=\s*Ka_tls\[t\]\s*;\s*"
        r"if\s*\(\s*!is_restricted\s*&&\s*nb_\s*>\s*0\s*\)\s*Kb_mat\s*\+=\s*Kb_tls\[t\]\s*;\s*"
        r"\}\s*"
        r"F_a\s*=\s*H_core_\s*\+\s*J_mat\s*-\s*Ka_mat\s*;",
        re.DOTALL,
    )

    replacement = f"""{marker}
    Eigen::MatrixXd Ka_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd Kb_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
    for (int t = 0; t < max_threads; ++t) {{
        Ka_mat.noalias() += Ka_tls[t];
        if (!is_restricted && nb_ > 0) Kb_mat.noalias() += Kb_tls[t];
    }}
    if (!is_idempotent_pa) {{
        Ka_mat = 0.5 * (Ka_mat + Ka_mat.transpose());
        if (!is_restricted && nb_ > 0) Kb_mat = 0.5 * (Kb_mat + Kb_mat.transpose());
    }}

    F_a = H_core_ + J_mat - Ka_mat;"""

    text, n = pat.subn(replacement, text, count=1)
    if n != 1:
        print(f"[ERROR] Gagal mencocokkan blok di {path}")
        return False

    path.write_text(text, encoding="utf-8")
    print(f"[OK]   Akumulasi & simetrisasi build_fock_fast selesai di {path}")
    return True


def patch_hpc_geom_rohf(root: Path) -> bool:
    path = root / "src/gradient/hpc_geom_opt.cc"
    if not path.exists():
        print(f"[ERROR] File tidak ditemukan: {path}")
        return False

    text = path.read_text(encoding="utf-8")
    marker = "// [INJECTED_FINAL_OPT_ROHF_PULAY_LAGRANGIAN]"
    if marker in text:
        print(f"[SKIP] {path} sudah terpatch ({marker})")
        return True

    pat = re.compile(
        r"const\s+Eigen::MatrixXd\s+Ca_o\s*=\s*scf_res\.C_alpha\.leftCols\(\s*na\s*\)\s*;\s*"
        r"W_eff_ao\s*=\s*Ca_o\s*\*\s*scf_res\.orbital_energies_alpha\.head\(\s*na\s*\)\.asDiagonal\(\)\s*\*\s*Ca_o\.transpose\(\)\s*;\s*"
        r"if\s*\(\s*is_rhf\s*\)\s*\{\s*"
        r"W_eff_ao\s*\*=\s*2\.0\s*;\s*"
        r"\}\s*else\s+if\s*\(\s*nb\s*>\s*0\s*\)\s*\{\s*"
        r"const\s+Eigen::MatrixXd\s+Cb_o\s*=\s*scf_res\.C_beta\.leftCols\(\s*nb\s*\)\s*;\s*"
        r"W_eff_ao\.noalias\(\)\s*\+=\s*Cb_o\s*\*\s*scf_res\.orbital_energies_beta\.head\(\s*nb\s*\)\.asDiagonal\(\)\s*\*\s*Cb_o\.transpose\(\)\s*;\s*"
        r"\}",
        re.DOTALL,
    )

    replacement = f"""const Eigen::MatrixXd Ca_o = scf_res.C_alpha.leftCols(na);
        if (ref == "rohf") {{
            {marker}
            const int n_c = nb;
            const int n_o = na - nb;
            const Eigen::MatrixXd& C = scf_res.C_alpha;
            Eigen::MatrixXd Fa_mo = C.transpose() * scf_res.F_alpha * C;
            Eigen::MatrixXd Fb_mo = C.transpose() * scf_res.F_beta * C;
            Eigen::MatrixXd W_mo = Eigen::MatrixXd::Zero(nbf, nbf);
            if (n_c > 0) {{
                W_mo.topLeftCorner(n_c, n_c) = Fa_mo.topLeftCorner(n_c, n_c) + Fb_mo.topLeftCorner(n_c, n_c);
            }}
            if (n_o > 0) {{
                W_mo.block(n_c, n_c, n_o, n_o) = Fa_mo.block(n_c, n_c, n_o, n_o);
            }}
            if (n_c > 0 && n_o > 0) {{
                Eigen::MatrixXd W_co = Fa_mo.block(0, n_c, n_c, n_o) + 0.5 * Fb_mo.block(0, n_c, n_c, n_o);
                W_mo.block(0, n_c, n_c, n_o) = W_co;
                W_mo.block(n_c, 0, n_o, n_c) = W_co.transpose();
            }}
            W_eff_ao = C * W_mo * C.transpose();
        }} else {{
            W_eff_ao = Ca_o * scf_res.orbital_energies_alpha.head(na).asDiagonal() * Ca_o.transpose();
            if (is_rhf) {{
                W_eff_ao *= 2.0;
            }} else if (nb > 0) {{
                const Eigen::MatrixXd Cb_o = scf_res.C_beta.leftCols(nb);
                W_eff_ao.noalias() += Cb_o * scf_res.orbital_energies_beta.head(nb).asDiagonal() * Cb_o.transpose();
            }}
        }}"""

    text, n = pat.subn(replacement, text, count=1)
    if n != 1:
        print(f"[ERROR] Gagal mencocokkan blok W_eff_ao di {path}")
        return False

    path.write_text(text, encoding="utf-8")
    print(f"[OK]   Lagrangian Pulay eksak DF-ROHF (blok cc, oo, co) terpasang di {path}")
    return True


def main() -> int:
    root = Path.cwd()
    print(f"=== MSHQC Remaining Patches Completion (root: {root}) ===\n")
    ok = all([
        patch_omp_gradient(root),
        patch_omp_fock(root),
        patch_hpc_geom_rohf(root),
    ])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())