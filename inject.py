#!/usr/bin/env python3
"""
Automated HPC Patch Injector for mshqc (Patches B, C, D, E)
Targets:
  - src/integrals/df_eri.cc
  - src/core/fock_builder.cc
  - include/mshqc/symmetry/molecule_sym.h
  - src/symmetry/molecule_sym.cc
  - src/mp3/mp3.cc
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def replace_between(
    content: str,
    start_anchor: str,
    end_anchor: str,
    replacement: str,
    include_end: bool = False,
    start_from: int = 0,
) -> tuple[str, int]:
    s_idx = content.find(start_anchor, start_from)
    if s_idx == -1:
        raise RuntimeError(f"Start anchor tidak ditemukan:\n{start_anchor[:120]}...")
    e_idx = content.find(end_anchor, s_idx + len(start_anchor))
    if e_idx == -1:
        raise RuntimeError(f"End anchor tidak ditemukan:\n{end_anchor[:120]}...")
    if include_end:
        e_idx += len(end_anchor)
    new_content = content[:s_idx] + replacement + content[e_idx:]
    return new_content, s_idx + len(replacement)


def ensure_include(content: str, header: str, after_header: str) -> str:
    inc_line = f"#include {header}"
    if inc_line in content:
        return content
    target = f"#include {after_header}"
    if target not in content:
        raise RuntimeError(f"Header referensi tidak ditemukan: {target}")
    return content.replace(target, f"{target}\n{inc_line}", 1)


# =============================================================================
# PATCH B: src/integrals/df_eri.cc
# =============================================================================
def patch_df_eri(content: str) -> tuple[str, bool]:
    if "aux_schwarz" in content and "buf_3c_sz" in content:
        return content, False

    content = ensure_include(content, "<algorithm>", "<omp.h>")
    content = ensure_include(content, "<cmath>", "<algorithm>")

    start_anchor = (
        "    // =====================================================================\n"
        "    // 2. REKONSTRUKSI MATRIKS METRIK J BERBASIS SIMD & OPENMP"
    )
    end_anchor = "    if (save_hdf5) {"

    replacement = r"""    int max_dim_p = 1, max_dim_a = 1;
    for (int sz : prim_sizes) max_dim_p = std::max(max_dim_p, sz);
    for (int sz : aux_sizes)  max_dim_a = std::max(max_dim_a, sz);

    const size_t buf_2c_sz = std::max<size_t>(64, static_cast<size_t>(max_dim_a) * max_dim_a);
    const size_t buf_3c_sz = std::max<size_t>(64, static_cast<size_t>(max_dim_p) * max_dim_p * max_dim_a);
    const size_t buf_4c_sz = std::max<size_t>(64, static_cast<size_t>(max_dim_p) * max_dim_p * max_dim_p * max_dim_p);

    // 2. Matriks Metrik 2-Pusat J_metric + batas Schwarz auxiliary Q_aux(R)
    std::cout << "  Evaluating 2-Center Metric Matrix...\n";
    Eigen::MatrixXd J_metric = Eigen::MatrixXd::Zero(n_aux_, n_aux_);
    #pragma omp parallel
    {
        std::vector<double> buffer(buf_2c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int R = 0; R < n_shells_aux; ++R) {
            int abs_R = n_shells_prim + R;
            int dim_R = aux_sizes[R];
            int bf_R_start = aux_starts[R];

            for (int S = 0; S <= R; ++S) {
                int abs_S = n_shells_prim + S;
                int dim_S = aux_sizes[S];
                int bf_S_start = aux_starts[S];

                if (!integrals_->compute_2c2e_block_raw(abs_R, abs_S, buffer.data())) continue;

                for (int s = 0; s < dim_S; ++s) {
                    int idx_S = bf_S_start + s;
                    int buffer_offset = dim_R * s;
                    for (int r = 0; r < dim_R; ++r) {
                        int idx_R = bf_R_start + r;
                        double val = buffer[r + buffer_offset];
                        J_metric(idx_R, idx_S) = val;
                        if (idx_R != idx_S) J_metric(idx_S, idx_R) = val;
                    }
                }
            }
        }
    }

    std::vector<double> aux_schwarz(n_shells_aux, 0.0);
    for (int R = 0; R < n_shells_aux; ++R) {
        double max_diag = 0.0;
        for (int r = 0; r < aux_sizes[R]; ++r) {
            max_diag = std::max(max_diag, std::abs(J_metric(aux_starts[R] + r, aux_starts[R] + r)));
        }
        aux_schwarz[R] = std::sqrt(max_diag);
    }

    // Batas Schwarz shell-pair primer Q_prim(i, j) = sqrt(max |(ij|ij)|)
    std::vector<double> prim_schwarz(static_cast<size_t>(n_shells_prim) * n_shells_prim, 0.0);
    #pragma omp parallel
    {
        std::vector<double> buffer(buf_4c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n_shells_prim; ++i) {
            int dim_i = prim_sizes[i];
            for (int j = 0; j <= i; ++j) {
                int dim_j = prim_sizes[j];
                if (!integrals_->compute_shell_block_raw(i, j, i, j, buffer.data())) continue;
                double max_v = 0.0;
                for (int p_i = 0; p_i < dim_i; ++p_i) {
                    for (int p_j = 0; p_j < dim_j; ++p_j) {
                        size_t idx = p_i + dim_i * (p_j + dim_j * (p_i + dim_i * p_j));
                        max_v = std::max(max_v, std::abs(buffer[idx]));
                    }
                }
                double s_val = std::sqrt(max_v);
                prim_schwarz[i * n_shells_prim + j] = s_val;
                prim_schwarz[j * n_shells_prim + i] = s_val;
            }
        }
    }

    // 3. Dekomposisi Eigen J^{-1/2} dengan OpenMP Tiled GEMM
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J_metric);
    const Eigen::VectorXd& evals = es.eigenvalues();
    const Eigen::MatrixXd& evecs = es.eigenvectors();

    double max_eval = (evals.size() > 0) ? std::max(1.0, evals.maxCoeff()) : 1.0;
    double rel_cond = (cutoff_ > 0.0 && cutoff_ <= 1e-10) ? cutoff_ : 1e-12;
    double eig_tol = std::max(1e-14, rel_cond * max_eval);

    int n_aux_rank = 0;
    for (int i = 0; i < n_aux_; ++i) {
        if (evals(i) >= eig_tol) n_aux_rank++;
    }

    Eigen::MatrixXd U_kept(n_aux_, n_aux_rank);
    Eigen::VectorXd inv_sqrt_evals(n_aux_rank);
    int col_idx = 0;
    for (int i = 0; i < n_aux_; ++i) {
        if (evals(i) >= eig_tol) {
            U_kept.col(col_idx) = evecs.col(i);
            inv_sqrt_evals(col_idx) = 1.0 / std::sqrt(evals(i));
            col_idx++;
        }
    }

    Eigen::MatrixXd U_scaled = U_kept * inv_sqrt_evals.asDiagonal();
    Eigen::MatrixXd J_inv_sqrt(n_aux_, n_aux_);
    constexpr int tile_j = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_aux_; r += tile_j) {
        int rows = std::min(tile_j, n_aux_ - r);
        J_inv_sqrt.middleRows(r, rows).noalias() = U_scaled.middleRows(r, rows) * U_kept.transpose();
    }

    // 4. Evaluasi Integral 3-Pusat dengan Schwarz Screening (tol = 1e-14)
    int n_pairs = n_primary_ * n_primary_;
    Eigen::MatrixXd V_raw = Eigen::MatrixXd::Zero(n_pairs, n_aux_);
    const double screen_tol = 1e-14;

    #pragma omp parallel
    {
        std::vector<double> buffer(buf_3c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n_shells_prim; ++i) {
            int bf_i_start = prim_starts[i];
            int dim_i = prim_sizes[i];

            for (int j = 0; j <= i; ++j) {
                double q_ij = prim_schwarz[i * n_shells_prim + j];
                if (q_ij < screen_tol) continue;

                int bf_j_start = prim_starts[j];
                int dim_j = prim_sizes[j];

                for (int R = 0; R < n_shells_aux; ++R) {
                    if (q_ij * aux_schwarz[R] < screen_tol) continue;

                    int abs_R = n_shells_prim + R;
                    int dim_R = aux_sizes[R];
                    int aux_start = aux_starts[R];

                    if (!integrals_->compute_3c2e_block_raw(i, j, abs_R, buffer.data())) continue;

                    for (int r = 0; r < dim_R; ++r) {
                        int P = aux_start + r;
                        for (int p_j = 0; p_j < dim_j; ++p_j) {
                            int base_r1 = bf_j_start + p_j;
                            int base_r2 = (bf_j_start + p_j) * n_primary_;
                            int buffer_offset = dim_i * (p_j + dim_j * r);

                            for (int p_i = 0; p_i < dim_i; ++p_i) {
                                double val = buffer[p_i + buffer_offset];
                                int r1 = (bf_i_start + p_i) * n_primary_ + base_r1;
                                int r2 = base_r2 + (bf_i_start + p_i);
                                V_raw(r1, P) = val;
                                if (r1 != r2) V_raw(r2, P) = val;
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Kontraksi OpenMP-Tiled BLAS-3 DGEMM: B = (mn|P) * J^{-1/2}_{PQ}
    B_mat_.resize(n_pairs, n_aux_);
    constexpr int tile_b = 256;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_pairs; r += tile_b) {
        int rows = std::min(tile_b, n_pairs - r);
        B_mat_.middleRows(r, rows).noalias() = V_raw.middleRows(r, rows) * J_inv_sqrt;
    }

"""
    content, _ = replace_between(content, start_anchor, end_anchor, replacement)
    return content, True


# =============================================================================
# PATCH C: src/core/fock_builder.cc
# =============================================================================
def patch_fock_builder(content: str) -> tuple[str, bool]:
    if "max_quartet_sz" in content and "compute_shell_block_raw" in content:
        return content, False

    dup_block = (
        "    if (dP_max < 1e-15) {\n"
        "        F_alpha = H_core_ + G_alpha_accum_;\n"
        "        if (!is_rhf) F_beta  = H_core_ + G_beta_accum_;\n"
        "        return;\n"
        "    }\n\n"
        "    P_alpha_old_ = P_alpha; \n"
        "    if (!is_rhf) P_beta_old_ = P_beta;\n"
    )
    clean_block = (
        "    if (dP_max < 1e-15) {\n"
        "        F_alpha = H_core_ + G_alpha_accum_;\n"
        "        if (!is_rhf) F_beta  = H_core_ + G_beta_accum_;\n"
        "        return;\n"
        "    }\n"
    )
    if dup_block in content:
        content = content.replace(dup_block, clean_block, 1)

    start_anchor = "    int n_threads = omp_get_max_threads();"
    end_anchor = "\n} // namespace mshqc"

    replacement = r"""    int max_shell_dim = 1;
    for (int s : shell_sizes) max_shell_dim = std::max(max_shell_dim, s);
    const size_t max_quartet_sz = std::max<size_t>(
        64, static_cast<size_t>(max_shell_dim) * max_shell_dim * max_shell_dim * max_shell_dim
    );

    auto process_shell_quartet = [&](Eigen::MatrixXd& Ga, Eigen::MatrixXd& Gb, double* buf,
                                     int M, int N, int P, int Q, double sym_weight) {
        if (!integrals_->compute_shell_block_raw(M, N, P, Q, buf)) return;

        int dM = shell_sizes[M], offM = shell_offsets[M];
        int dN = shell_sizes[N], offN = shell_offsets[N];
        int dP = shell_sizes[P], offP = shell_offsets[P];
        int dQ = shell_sizes[Q], offQ = shell_offsets[Q];

        double fac = sym_weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        const double* I_ptr = buf;
        for (int q = 0; q < dQ; ++q) {
            int sig = offQ + q;
            for (int p = 0; p < dP; ++p) {
                int lam = offP + p;
                double pt_ls_2 = (dP_a(lam, sig) + dP_b(lam, sig)) * 2.0;
                double j_ls_acc = 0.0;
                for (int n = 0; n < dN; ++n) {
                    int nu = offN + n;
                    double dPa_ns = dP_a(nu, sig), dPa_nl = dP_a(nu, lam);
                    double dPb_ns = dP_b(nu, sig), dPb_nl = dP_b(nu, lam);
                    for (int m = 0; m < dM; ++m) {
                        int mu = offM + m;
                        double val = *I_ptr++;
                        if (std::abs(val) < 1e-14) continue;

                        double v = val * fac;
                        double j_val = v * pt_ls_2;
                        Ga(mu, nu) += j_val;
                        Gb(mu, nu) += j_val;
                        j_ls_acc += v * (dP_a(mu, nu) + dP_b(mu, nu)) * 2.0;

                        Ga(mu, lam) -= v * dPa_ns;
                        Ga(nu, lam) -= v * dP_a(mu, sig);
                        Ga(mu, sig) -= v * dPa_nl;
                        Ga(nu, sig) -= v * dP_a(mu, lam);

                        Gb(mu, lam) -= v * dPb_ns;
                        Gb(nu, lam) -= v * dP_b(mu, sig);
                        Gb(mu, sig) -= v * dPb_nl;
                        Gb(nu, sig) -= v * dP_b(mu, lam);
                    }
                }
                Ga(lam, sig) += j_ls_acc;
                Gb(lam, sig) += j_ls_acc;
            }
        }
    };

    auto process_shell_quartet_rhf = [&](Eigen::MatrixXd& Ga, double* buf,
                                         int M, int N, int P, int Q, double sym_weight) {
        if (!integrals_->compute_shell_block_raw(M, N, P, Q, buf)) return;

        int dM = shell_sizes[M], offM = shell_offsets[M];
        int dN = shell_sizes[N], offN = shell_offsets[N];
        int dP = shell_sizes[P], offP = shell_offsets[P];
        int dQ = shell_sizes[Q], offQ = shell_offsets[Q];

        double fac = sym_weight;
        if (M == N) fac *= 0.5;
        if (P == Q) fac *= 0.5;
        if (M == P && N == Q) fac *= 0.5;

        const double* I_ptr = buf;
        for (int q = 0; q < dQ; ++q) {
            int sig = offQ + q;
            for (int p = 0; p < dP; ++p) {
                int lam = offP + p;
                double pt_ls_4 = dP_a(lam, sig) * 4.0;
                double j_ls_acc = 0.0;
                for (int n = 0; n < dN; ++n) {
                    int nu = offN + n;
                    double dPa_ns = dP_a(nu, sig), dPa_nl = dP_a(nu, lam);
                    for (int m = 0; m < dM; ++m) {
                        int mu = offM + m;
                        double val = *I_ptr++;
                        if (std::abs(val) < 1e-14) continue;

                        double v = val * fac;
                        Ga(mu, nu)  += v * pt_ls_4;
                        j_ls_acc    += v * dP_a(mu, nu) * 4.0;

                        Ga(mu, lam) -= v * dPa_ns;
                        Ga(nu, lam) -= v * dP_a(mu, sig);
                        Ga(mu, sig) -= v * dPa_nl;
                        Ga(nu, sig) -= v * dP_a(mu, lam);
                    }
                }
                Ga(lam, sig) += j_ls_acc;
            }
        }
    };

    Eigen::MatrixXd dGa = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    Eigen::MatrixXd dGb = is_rhf ? Eigen::MatrixXd() : Eigen::MatrixXd::Zero(nbasis_, nbasis_);

    #pragma omp parallel
    {
        std::vector<double> buf(max_quartet_sz);
        Eigen::MatrixXd Ga_loc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Gb_loc = is_rhf ? Eigen::MatrixXd() : Eigen::MatrixXd::Zero(nbasis_, nbasis_);

        if (petite_list_ && !petite_list_->get_unique_quartets().empty()) {
            const auto& uq = petite_list_->get_unique_quartets();
            #pragma omp for schedule(dynamic, 16) nowait
            for (size_t i = 0; i < uq.size(); ++i) {
                int M = uq[i].M, N = uq[i].N, P = uq[i].P, Q = uq[i].Q;
                double bound = schwarz_(M, N) * schwarz_(P, Q);
                if (bound * dP_max < threshold) continue;
                double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                if (bound * local_P_max < threshold) continue;

                if (is_rhf) process_shell_quartet_rhf(Ga_loc, buf.data(), M, N, P, Q, uq[i].weight);
                else        process_shell_quartet(Ga_loc, Gb_loc, buf.data(), M, N, P, Q, uq[i].weight);
            }
        } else {
            #pragma omp for schedule(dynamic, 1) nowait
            for (int M = 0; M < nshells; ++M) {
                for (int N = 0; N <= M; ++N) {
                    double bound_MN = schwarz_(M, N);
                    if (bound_MN * schwarz_max * dP_max < threshold) continue;
                    for (int P = 0; P <= M; ++P) {
                        int Q_max = (M == P) ? N : P;
                        for (int Q = 0; Q <= Q_max; ++Q) {
                            double bound = bound_MN * schwarz_(P, Q);
                            double local_P_max = std::max({P_max(M, N), P_max(P, Q), P_max(M, P), P_max(M, Q), P_max(N, P), P_max(N, Q)});
                            if (bound * local_P_max < threshold) continue;

                            if (is_rhf) process_shell_quartet_rhf(Ga_loc, buf.data(), M, N, P, Q, 1.0);
                            else        process_shell_quartet(Ga_loc, Gb_loc, buf.data(), M, N, P, Q, 1.0);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            dGa.noalias() += Ga_loc;
            if (!is_rhf) dGb.noalias() += Gb_loc;
        }
    }

    G_alpha_accum_.noalias() += dGa + dGa.transpose();
    F_alpha = H_core_ + G_alpha_accum_;

    if (!is_rhf) {
        G_beta_accum_.noalias() += dGb + dGb.transpose();
        F_beta = H_core_ + G_beta_accum_;
    }
}"""
    content, _ = replace_between(content, start_anchor, end_anchor, replacement)
    return content, True


# =============================================================================
# PATCH D1: include/mshqc/symmetry/molecule_sym.h
# =============================================================================
def patch_molecule_sym_h(content: str) -> tuple[str, bool]:
    if "ao_perm_" in content and "ao_phase_" in content:
        return content, False

    pattern = re.compile(r"([ \t]*std::vector<Eigen::MatrixXd>\s+R_ao_;)")
    if not pattern.search(content):
        raise RuntimeError("Deklarasi member R_ao_ tidak ditemukan di molecule_sym.h")

    content = pattern.sub(
        r"\1\n    std::vector<std::vector<int>> ao_perm_;\n    std::vector<std::vector<double>> ao_phase_;",
        content,
        count=1,
    )
    return content, True


# =============================================================================
# PATCH D2: src/symmetry/molecule_sym.cc
# =============================================================================
def patch_molecule_sym_cc(content: str) -> tuple[str, bool]:
    if "ao_perm_" in content and "ao_phase_" in content:
        return content, False

    content = ensure_include(content, "<vector>", "<cmath>")

    start_anchor = "void BasisSymmetrizer::build_map_and_matrices() {"
    end_anchor = "\n} // namespace mshqc"

    replacement = r"""void BasisSymmetrizer::build_map_and_matrices() {
    int nshells = basis_.n_shells();
    int nbasis = basis_.n_basis_functions();
    const auto& ops = pg_.get_operations();
    int n_ops = static_cast<int>(ops.size());

    shell_map_.assign(n_ops, std::vector<int>(nshells, -1));
    R_ao_.assign(n_ops, Eigen::MatrixXd::Zero(nbasis, nbasis));
    ao_perm_.assign(n_ops, std::vector<int>(nbasis, -1));
    ao_phase_.assign(n_ops, std::vector<double>(nbasis, 0.0));

    auto map = basis_.shell_to_basis_function_map();

    for (int k = 0; k < n_ops; ++k) {
        const auto& R = ops[k].matrix;
        double sx = (R(0,0) > 0.1) ? 1.0 : -1.0;
        double sy = (R(1,1) > 0.1) ? 1.0 : -1.0;
        double sz = (R(2,2) > 0.1) ? 1.0 : -1.0;

        for (int i = 0; i < nshells; ++i) {
            Eigen::Vector3d p_i(basis_.shell(i).position()[0], basis_.shell(i).position()[1], basis_.shell(i).position()[2]);
            Eigen::Vector3d new_pos = R * p_i;
            int l_target = basis_.shell(i).l();
            
            int n_prev = 0;
            for (int x = 0; x < i; ++x) {
                if (basis_.shell(x).l() == l_target) {
                    Eigen::Vector3d px(basis_.shell(x).position()[0], basis_.shell(x).position()[1], basis_.shell(x).position()[2]);
                    if ((px - p_i).norm() < 1e-3) n_prev++;
                }
            }

            int n_found = 0;
            int target_idx = -1;
            for (int j = 0; j < nshells; ++j) {
                if (basis_.shell(j).l() != l_target) continue;
                Eigen::Vector3d pj(basis_.shell(j).position()[0], basis_.shell(j).position()[1], basis_.shell(j).position()[2]);
                if ((pj - new_pos).norm() < 1e-3) {
                    if (n_found == n_prev) { target_idx = j; break; }
                    n_found++;
                }
            }
            
            shell_map_[k][i] = target_idx;

            if (target_idx != -1) {
                int start_i = map[i];
                int start_j = map[target_idx];
                int dim = basis_.shell(i).is_spherical() ? (2*l_target + 1) : ((l_target+1)*(l_target+2)/2);
                
                for (int d = 0; d < dim; ++d) {
                    double phase = get_universal_phase(basis_.shell(i).is_spherical(), l_target, d, sx, sy, sz);
                    R_ao_[k](start_j + d, start_i + d) = phase;
                    ao_perm_[k][start_i + d] = start_j + d;
                    ao_phase_[k][start_i + d] = phase;
                }
            }
        }
    }
}

void BasisSymmetrizer::symmetrize(Eigen::MatrixXd& F) const {
    const int n_ops = static_cast<int>(ao_perm_.size());
    if (n_ops == 0) return;

    const int n = static_cast<int>(F.rows());
    Eigen::MatrixXd F_sym = Eigen::MatrixXd::Zero(n, n);

    for (int k = 0; k < n_ops; ++k) {
        const auto& perm = ao_perm_[k];
        const auto& ph   = ao_phase_[k];
        for (int q = 0; q < n; ++q) {
            const int rq = perm[q];
            if (rq < 0) continue;
            const double ph_q = ph[q];
            const double* F_col = F.col(rq).data();
            double* dst_col = F_sym.col(q).data();
            for (int p = 0; p < n; ++p) {
                const int rp = perm[p];
                if (rp >= 0) dst_col[p] += ph[p] * ph_q * F_col[rp];
            }
        }
    }
    F.noalias() = F_sym / static_cast<double>(n_ops);
}

std::vector<int> BasisSymmetrizer::assign_mo_irreps(const Eigen::MatrixXd& C, double threshold) const {
    (void)threshold;
    const int nbf = static_cast<int>(C.rows());
    const int n_mo = static_cast<int>(C.cols());
    const int n_ops = static_cast<int>(ao_perm_.size());
    std::vector<int> irreps(n_mo, 0);

    for (int mo = 0; mo < n_mo; ++mo) {
        const double* c_ptr = C.col(mo).data();
        double norm_c = C.col(mo).norm();
        int irrep_id = 0;

        for (int k = 0; k < n_ops; ++k) {
            const auto& perm = ao_perm_[k];
            const auto& ph   = ao_phase_[k];
            double chi = 0.0;
            double norm_t_sq = 0.0;
            for (int p = 0; p < nbf; ++p) {
                const int rp = perm[p];
                if (rp >= 0) {
                    const double val = ph[p] * c_ptr[p];
                    chi += c_ptr[rp] * val;
                    norm_t_sq += val * val;
                }
            }
            double denom = norm_c * std::sqrt(norm_t_sq);
            if (denom > 1e-10 && (chi / denom) < -0.5) {
                irrep_id |= (1 << k);
            }
        }

        int mapped_id = 0;
        if (irrep_id & (1 << 1)) mapped_id ^= 1;
        if (irrep_id & (1 << 2)) mapped_id ^= 2;
        if (irrep_id & (1 << 4)) mapped_id ^= 4;
        irreps[mo] = mapped_id;
    }
    return irreps;
}"""
    content, _ = replace_between(content, start_anchor, end_anchor, replacement)
    return content, True


# =============================================================================
# PATCH E: src/mp3/mp3.cc
# =============================================================================
def patch_mp3(content: str) -> tuple[str, bool]:
    if "make_V_from_B_inplace" not in content and "contract_vvvv_pairs_df(B_vv," in content:
        return content, False

    # 1. RMP3::compute() - Hapus make_V_from_B_inplace
    content, pos = replace_between(
        content,
        "    // Bypass TBLIS/MKL thread limits with Tiled Level-3 BLAS DGEMM & layout-respecting mapping\n"
        "    auto make_V_from_B_inplace =",
        "    Eigen::MatrixXd B_oo, B_ov, B_vv;",
        "",
    )

    # 2. RMP3::compute() - Ladder vvvv (cabang non-exact)
    rmp3_vvvv_old = (
        "        #pragma omp parallel\n"
        "        {\n"
        "            std::vector<Eigen::MatrixXd> W_loc(n_pairs_sym, Eigen::MatrixXd::Zero(nv_a_, nv_a_));\n"
        "            Eigen::MatrixXd X_tmp(nv_a_, nv_a_);\n"
        "            #pragma omp for schedule(static) nowait\n"
        "            for (int P = 0; P < n_aux_; ++P) {\n"
        "                Eigen::Map<const Eigen::MatrixXd> B_P(B_vv.col(P).data(), nv_a_, nv_a_);\n"
        "                for (int k = 0; k < n_pairs_sym; ++k) {\n"
        "                    X_tmp.noalias() = T_pairs[k] * B_P;\n"
        "                    W_loc[k].noalias() += B_P.transpose() * X_tmp;\n"
        "                }\n"
        "            }\n"
        "            #pragma omp critical\n"
        "            {\n"
        "                for (int k = 0; k < n_pairs_sym; ++k) W_pairs[k] += W_loc[k];\n"
        "            }\n"
        "        }"
    )
    if rmp3_vvvv_old not in content:
        raise RuntimeError("Blok RMP3 vvvv ladder tidak ditemukan")
    content = content.replace(
        rmp3_vvvv_old,
        "        contract_vvvv_pairs_df(B_vv, nv_a_, B_vv, nv_a_, n_aux_, T_pairs, W_pairs);",
        1,
    )

    # 3. RMP3::compute() - Ladder (oooo) + Ring (ovov & oovv)
    rmp3_ring_new = r"""    // 2 & 3. Ladder term (oooo) + Ring terms (ovov & oovv)
    if (config_.eri_method == "exact") {
        auto half_oo = ERITransformer::transform_half_bra(*eri_ao_ptr, Co, Co, nbf_, no_a_, no_a_);
        Eigen::Tensor<double, 4> V      = ERITransformer::transform_finish_ket(half_oo, Co, Co, nbf_, no_a_, no_a_, no_a_, no_a_);
        Eigen::Tensor<double, 4> V_oovv = ERITransformer::transform_finish_ket(half_oo, Cv, Cv, nbf_, no_a_, no_a_, nv_a_, nv_a_);
        half_oo.resize(0, 0, 0, 0);
        Eigen::Tensor<double, 4> V_ovov = ERITransformer::transform_custom(*eri_ao_ptr, Co, Cv, Co, Cv, nbf_, no_a_, nv_a_, no_a_, nv_a_);

        TBLIS_VIEW_4D(t_V, V, no_a_, no_a_, no_a_, no_a_);
        TBLIS_VIEW_4D(t_Vovov, V_ovov, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_Voovv, V_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>( 1.0, t_T,     "mnab", t_V,     "minj", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_Vovov, "iakc", t_T,     "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_T,     "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T,     "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T,     "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T,     "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T,     "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T,     "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T,     "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");
    } else {
        Eigen::MatrixXd M_oooo = parallel_gemm_abt_mp3(B_oo, B_oo);
        Eigen::MatrixXd M_ovov = parallel_gemm_abt_mp3(B_ov, B_ov);
        Eigen::MatrixXd M_oovv = parallel_gemm_abt_mp3(B_oo, B_vv);

        auto t_V     = make_mat_view_4d_mp3(M_oooo, no_a_, no_a_, no_a_, no_a_);
        auto t_Vovov = make_mat_view_4d_mp3(M_ovov, no_a_, nv_a_, no_a_, nv_a_);
        auto t_Voovv = make_mat_view_4d_mp3(M_oovv, no_a_, no_a_, nv_a_, nv_a_);

        tblis::mult<double>( 1.0, t_T,     "mnab", t_V,     "minj", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_Vovov, "iakc", t_T,     "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>( 2.0, t_T,     "ikac", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T,     "ikca", t_Vovov, "kcjb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Vovov, "iakc", t_T,     "kjbc", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikac", t_T,     "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T,     "ikac", t_Voovv, "kjcb", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_Voovv, "ikbc", t_T,     "kjac", 1.0, t_W, "ijab");
        tblis::mult<double>(-1.0, t_T,     "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");
    }

"""
    content, pos = replace_between(
        content,
        "    // 2 & 3. Ladder term (oooo) + Ring terms (ovov & oovv) dengan Shared Half-Bra (Co, Co)",
        "    double e_mp3 = 0.0;",
        rmp3_ring_new,
    )

    # 4. UMP3::compute() - Hapus make_V_from_B_inplace
    content, pos = replace_between(
        content,
        "    auto make_V_from_B_inplace =",
        "    Eigen::MatrixXd B_oo_a, B_ov_a, B_vv_a;",
        "",
        start_from=pos,
    )

    # 5. UMP3::compute() - Ladder vvvv (AA, BB, AB)
    ump3_vvvv_aa = (
        "            #pragma omp parallel\n"
        "            {\n"
        "                std::vector<Eigen::MatrixXd> W_loc(n_pairs_aa, Eigen::MatrixXd::Zero(nv_a_, nv_a_));\n"
        "                Eigen::MatrixXd X_tmp(nv_a_, nv_a_);\n"
        "                #pragma omp for schedule(static) nowait\n"
        "                for (int P = 0; P < n_aux_; ++P) {\n"
        "                    Eigen::Map<const Eigen::MatrixXd> B_P(B_vv_a.col(P).data(), nv_a_, nv_a_);\n"
        "                    for (int k = 0; k < n_pairs_aa; ++k) {\n"
        "                        X_tmp.noalias() = T_aa_pairs[k] * B_P;\n"
        "                        W_loc[k].noalias() += B_P.transpose() * X_tmp;\n"
        "                    }\n"
        "                }\n"
        "                #pragma omp critical\n"
        "                {\n"
        "                    for (int k = 0; k < n_pairs_aa; ++k) W_aa_pairs[k] += W_loc[k];\n"
        "                }\n"
        "            }"
    )
    ump3_vvvv_bb = (
        "                #pragma omp parallel\n"
        "                {\n"
        "                    std::vector<Eigen::MatrixXd> W_loc(n_pairs_bb, Eigen::MatrixXd::Zero(nv_b_, nv_b_));\n"
        "                    Eigen::MatrixXd X_tmp(nv_b_, nv_b_);\n"
        "                    #pragma omp for schedule(static) nowait\n"
        "                    for (int P = 0; P < n_aux_; ++P) {\n"
        "                        Eigen::Map<const Eigen::MatrixXd> B_P(B_vv_b.col(P).data(), nv_b_, nv_b_);\n"
        "                        for (int k = 0; k < n_pairs_bb; ++k) {\n"
        "                            X_tmp.noalias() = T_bb_pairs[k] * B_P;\n"
        "                            W_loc[k].noalias() += B_P.transpose() * X_tmp;\n"
        "                        }\n"
        "                    }\n"
        "                    #pragma omp critical\n"
        "                    {\n"
        "                        for (int k = 0; k < n_pairs_bb; ++k) W_bb_pairs[k] += W_loc[k];\n"
        "                    }\n"
        "                }"
    )
    ump3_vvvv_ab = (
        "            #pragma omp parallel\n"
        "            {\n"
        "                std::vector<Eigen::MatrixXd> W_loc(n_pairs_ab, Eigen::MatrixXd::Zero(nv_a_, nv_b_));\n"
        "                Eigen::MatrixXd X_tmp(nv_a_, nv_b_);\n"
        "                #pragma omp for schedule(static) nowait\n"
        "                for (int P = 0; P < n_aux_; ++P) {\n"
        "                    Eigen::Map<const Eigen::MatrixXd> B_Pa(B_vv_a.col(P).data(), nv_a_, nv_a_);\n"
        "                    Eigen::Map<const Eigen::MatrixXd> B_Pb(B_vv_b.col(P).data(), nv_b_, nv_b_);\n"
        "                    for (int k = 0; k < n_pairs_ab; ++k) {\n"
        "                        X_tmp.noalias() = T_ab_pairs[k] * B_Pb;\n"
        "                        W_loc[k].noalias() += B_Pa.transpose() * X_tmp;\n"
        "                    }\n"
        "                }\n"
        "                #pragma omp critical\n"
        "                {\n"
        "                    for (int k = 0; k < n_pairs_ab; ++k) W_ab_pairs[k] += W_loc[k];\n"
        "                }\n"
        "            }"
    )

    for old_blk, new_blk in [
        (ump3_vvvv_aa, "            contract_vvvv_pairs_df(B_vv_a, nv_a_, B_vv_a, nv_a_, n_aux_, T_aa_pairs, W_aa_pairs);"),
        (ump3_vvvv_bb, "                contract_vvvv_pairs_df(B_vv_b, nv_b_, B_vv_b, nv_b_, n_aux_, T_bb_pairs, W_bb_pairs);"),
        (ump3_vvvv_ab, "            contract_vvvv_pairs_df(B_vv_a, nv_a_, B_vv_b, nv_b_, n_aux_, T_ab_pairs, W_ab_pairs);"),
    ]:
        if old_blk not in content:
            raise RuntimeError(f"Blok UMP3 vvvv tidak ditemukan:\n{old_blk[:100]}...")
        content = content.replace(old_blk, new_blk, 1)

    # 6. UMP3::compute() - Ladder (oooo) & Ring (ovov & oovv)
    ump3_ring_new = r"""    // =========================================================================
    // 2 & 3. TAHAP LADDER (oooo) & RING (ovov & oovv)
    // =========================================================================
    if (config_.eri_method == "exact") {
        auto half_oo_a = ERITransformer::transform_half_bra(*eri_ao_ptr, Cao, Cao, nbf_, no_a_, no_a_);
        auto half_ov_a = ERITransformer::transform_half_bra(*eri_ao_ptr, Cao, Cav, nbf_, no_a_, nv_a_);
        Eigen::Tensor<double, 4> half_oo_b;
        if (no_b_ > 0 && nv_b_ > 0) {
            half_oo_b = ERITransformer::transform_half_bra(*eri_ao_ptr, Cbo, Cbo, nbf_, no_b_, no_b_);
        }

        // --- 2. LADDER (oooo) EXACT ---
        if (no_a_ > 1) {
            auto Vaa = ERITransformer::transform_finish_ket(half_oo_a, Cao, Cao, nbf_, no_a_, no_a_, no_a_, no_a_);
            TBLIS_VIEW_4D(t_Vaa, Vaa, no_a_, no_a_, no_a_, no_a_);
            Waa.setZero();
            tblis::mult<double>( 1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
            e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            if (no_b_ > 1) {
                auto Vbb = ERITransformer::transform_finish_ket(half_oo_b, Cbo, Cbo, nbf_, no_b_, no_b_, no_b_, no_b_);
                TBLIS_VIEW_4D(t_Vbb, Vbb, no_b_, no_b_, no_b_, no_b_);
                Wbb.setZero();
                tblis::mult<double>( 1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
                e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);
            }

            auto Vab = ERITransformer::transform_finish_ket(half_oo_a, Cbo, Cbo, nbf_, no_a_, no_a_, no_b_, no_b_);
            TBLIS_VIEW_4D(t_Vab, Vab, no_a_, no_a_, no_b_, no_b_);
            Wab.setZero();
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        }

        // --- 3. RING (ovov & oovv) EXACT ---
        auto ovov_aa = ERITransformer::transform_finish_ket(half_ov_a, Cao, Cav, nbf_, no_a_, nv_a_, no_a_, nv_a_);
        auto oovv_aa = ERITransformer::transform_finish_ket(half_oo_a, Cav, Cav, nbf_, no_a_, no_a_, nv_a_, nv_a_);
        TBLIS_VIEW_4D(t_ovov_aa, ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        TBLIS_VIEW_4D(t_oovv_aa, oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        Waa.setZero();
        if (no_a_ > 1) {
            tblis::mult<double>( 1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa, "ikca", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            auto ovov_bb    = ERITransformer::transform_custom(*eri_ao_ptr, Cbo, Cbv, Cbo, Cbv, nbf_, no_b_, nv_b_, no_b_, nv_b_);
            auto oovv_bb    = ERITransformer::transform_finish_ket(half_oo_b, Cbv, Cbv, nbf_, no_b_, no_b_, nv_b_, nv_b_);
            auto ovov_ab    = ERITransformer::transform_finish_ket(half_ov_a, Cbo, Cbv, nbf_, no_a_, nv_a_, no_b_, nv_b_);
            auto oovv_ab_ex = ERITransformer::transform_finish_ket(half_oo_a, Cbv, Cbv, nbf_, no_a_, no_a_, nv_b_, nv_b_);
            auto oovv_ba_ex = ERITransformer::transform_finish_ket(half_oo_b, Cav, Cav, nbf_, no_b_, no_b_, nv_a_, nv_a_);
            half_oo_a.resize(0, 0, 0, 0);
            half_oo_b.resize(0, 0, 0, 0);
            half_ov_a.resize(0, 0, 0, 0);

            TBLIS_VIEW_4D(t_ovov_bb, ovov_bb, no_b_, nv_b_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_bb, oovv_bb, no_b_, no_b_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_ovov_ab, ovov_ab, no_a_, nv_a_, no_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_ab_ex, oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            TBLIS_VIEW_4D(t_oovv_ba_ex, oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            if (no_a_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

            Wbb.setZero();
            if (no_b_ > 1) {
                tblis::mult<double>( 1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_oovv_bb, "ikca", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>( 1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab");
                e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);
            }

            Wab.setZero();
            tblis::mult<double>( 1.0, t_ovov_aa,    "iakc", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa,    "ikca", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>( 1.0, t_Tab,        "ikac", t_ovov_bb,    "kcjb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikac", t_oovv_bb,    "kjcb", 1.0, t_Wab, "ijab");
            if (no_a_ > 1) tblis::mult<double>(1.0, t_Taa,     "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
            if (no_b_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tbb,     "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_ab_ex, "ikbc", t_Tab,          "kjac", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        } else {
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        }
    } else {
        // --- 2. LADDER (oooo) DF/CHOLESKY (ZERO-COPY) ---
        if (no_a_ > 1) {
            Eigen::MatrixXd M_oooo_aa = parallel_gemm_abt_mp3(B_oo_a, B_oo_a);
            auto t_Vaa = make_mat_view_4d_mp3(M_oooo_aa, no_a_, no_a_, no_a_, no_a_);
            Waa.setZero();
            tblis::mult<double>( 1.0, t_Taa, "mnab", t_Vaa, "minj", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_Taa, "mnab", t_Vaa, "mjni", 1.0, t_Waa, "ijab");
            e3_aa += 0.125 * tensor_dot(t2_aa_, Waa);
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            if (no_b_ > 1) {
                Eigen::MatrixXd M_oooo_bb = parallel_gemm_abt_mp3(B_oo_b, B_oo_b);
                auto t_Vbb = make_mat_view_4d_mp3(M_oooo_bb, no_b_, no_b_, no_b_, no_b_);
                Wbb.setZero();
                tblis::mult<double>( 1.0, t_Tbb, "mnab", t_Vbb, "minj", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_Tbb, "mnab", t_Vbb, "mjni", 1.0, t_Wbb, "ijab");
                e3_bb += 0.125 * tensor_dot(t2_bb_, Wbb);
            }

            Eigen::MatrixXd M_oooo_ab = parallel_gemm_abt_mp3(B_oo_a, B_oo_b);
            auto t_Vab = make_mat_view_4d_mp3(M_oooo_ab, no_a_, no_a_, no_b_, no_b_);
            Wab.setZero();
            tblis::mult<double>(1.0, t_Tab, "mnab", t_Vab, "minj", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        }

        // --- 3. RING (ovov & oovv) DF/CHOLESKY (ZERO-COPY) ---
        Eigen::MatrixXd M_ovov_aa = parallel_gemm_abt_mp3(B_ov_a, B_ov_a);
        Eigen::MatrixXd M_oovv_aa = parallel_gemm_abt_mp3(B_oo_a, B_vv_a);
        auto t_ovov_aa = make_mat_view_4d_mp3(M_ovov_aa, no_a_, nv_a_, no_a_, nv_a_);
        auto t_oovv_aa = make_mat_view_4d_mp3(M_oovv_aa, no_a_, no_a_, nv_a_, nv_a_);

        Waa.setZero();
        if (no_a_ > 1) {
            tblis::mult<double>( 1.0, t_ovov_aa, "iakc", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa, "ikca", t_Taa, "kjcb", 1.0, t_Waa, "ijab");
        }

        if (no_b_ > 0 && nv_b_ > 0) {
            Eigen::MatrixXd M_ovov_bb    = parallel_gemm_abt_mp3(B_ov_b, B_ov_b);
            Eigen::MatrixXd M_oovv_bb    = parallel_gemm_abt_mp3(B_oo_b, B_vv_b);
            Eigen::MatrixXd M_ovov_ab    = parallel_gemm_abt_mp3(B_ov_a, B_ov_b);
            Eigen::MatrixXd M_oovv_ab_ex = parallel_gemm_abt_mp3(B_oo_a, B_vv_b);
            Eigen::MatrixXd M_oovv_ba_ex = parallel_gemm_abt_mp3(B_oo_b, B_vv_a);

            auto t_ovov_bb    = make_mat_view_4d_mp3(M_ovov_bb,    no_b_, nv_b_, no_b_, nv_b_);
            auto t_oovv_bb    = make_mat_view_4d_mp3(M_oovv_bb,    no_b_, no_b_, nv_b_, nv_b_);
            auto t_ovov_ab    = make_mat_view_4d_mp3(M_ovov_ab,    no_a_, nv_a_, no_b_, nv_b_);
            auto t_oovv_ab_ex = make_mat_view_4d_mp3(M_oovv_ab_ex, no_a_, no_a_, nv_b_, nv_b_);
            auto t_oovv_ba_ex = make_mat_view_4d_mp3(M_oovv_ba_ex, no_b_, no_b_, nv_a_, nv_a_);

            if (no_a_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tab, "jkbc", 1.0, t_Waa, "ijab");
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);

            Wbb.setZero();
            if (no_b_ > 1) {
                tblis::mult<double>( 1.0, t_ovov_bb, "iakc", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>(-1.0, t_oovv_bb, "ikca", t_Tbb, "kjcb", 1.0, t_Wbb, "ijab");
                tblis::mult<double>( 1.0, t_ovov_ab, "kcia", t_Tab, "kjcb", 1.0, t_Wbb, "ijab");
                e3_bb += 1.0 * tensor_dot(t2_bb_, Wbb);
            }

            Wab.setZero();
            tblis::mult<double>( 1.0, t_ovov_aa,    "iakc", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_aa,    "ikca", t_Tab,          "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>( 1.0, t_Tab,        "ikac", t_ovov_bb,    "kcjb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikac", t_oovv_bb,    "kjcb", 1.0, t_Wab, "ijab");
            if (no_a_ > 1) tblis::mult<double>(1.0, t_Taa,     "ikac", t_ovov_ab, "kcjb", 1.0, t_Wab, "ijab");
            if (no_b_ > 1) tblis::mult<double>(1.0, t_ovov_ab, "iakc", t_Tbb,     "kjcb", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_oovv_ab_ex, "ikbc", t_Tab,          "kjac", 1.0, t_Wab, "ijab");
            tblis::mult<double>(-1.0, t_Tab,        "ikcb", t_oovv_ba_ex, "jkac", 1.0, t_Wab, "ijab");
            e3_ab += 1.0 * tensor_dot(t2_ab_, Wab);
        } else {
            e3_aa += 1.0 * tensor_dot(t2_aa_, Waa);
        }
    }

"""
    content, pos = replace_between(
        content,
        "    // =========================================================================\n"
        "    // 2 & 3. TAHAP LADDER (oooo) & RING (ovov & oovv) DENGAN SHARED HALF-BRA",
        "    MP3Result res;",
        ump3_ring_new,
        start_from=pos,
    )

    # 7. OMP3::compute_mp3_correction() - 4 blok ladder vvvv DF/Cholesky
    omp3_vvvv_blocks = [
        (
            "        #pragma omp parallel\n"
            "        {\n"
            "            std::vector<Eigen::MatrixXd> W_loc(n_pairs_sym, Eigen::MatrixXd::Zero(va_, va_));\n"
            "            Eigen::MatrixXd X_tmp(va_, va_);\n\n"
            "            #pragma omp for schedule(static) nowait\n"
            "            for (int P = 0; P < n_aux; ++P) {\n"
            "                Eigen::Map<const Eigen::MatrixXd> B_P(B_ab_a.col(P).data(), va_, va_);\n"
            "                for (int k = 0; k < n_pairs_sym; ++k) {\n"
            "                    X_tmp.noalias() = T_pairs[k] * B_P;\n"
            "                    W_loc[k].noalias() += B_P.transpose() * X_tmp;\n"
            "                }\n"
            "            }\n\n"
            "            #pragma omp critical\n"
            "            {\n"
            "                for (int k = 0; k < n_pairs_sym; ++k) W_pairs[k] += W_loc[k];\n"
            "            }\n"
            "        }",
            "        contract_vvvv_pairs_df(B_ab_a, va_, B_ab_a, va_, n_aux, T_pairs, W_pairs);",
        ),
        (
            "            #pragma omp parallel for schedule(static)\n"
            "            for (int k = 0; k < n_pairs_aa; ++k) {\n"
            "                Eigen::MatrixXd X_tmp(va_, va_);\n"
            "                for (int P = 0; P < n_aux; ++P) {\n"
            "                    Eigen::Map<const Eigen::MatrixXd> B_P(B_ab_a.col(P).data(), va_, va_);\n"
            "                    X_tmp.noalias() = T_aa_pairs[k] * B_P;\n"
            "                    W_aa_pairs[k].noalias() += B_P.transpose() * X_tmp;\n"
            "                }\n"
            "            }",
            "            contract_vvvv_pairs_df(B_ab_a, va_, B_ab_a, va_, n_aux, T_aa_pairs, W_aa_pairs);",
        ),
        (
            "                #pragma omp parallel for schedule(static)\n"
            "                for (int k = 0; k < n_pairs_bb; ++k) {\n"
            "                    Eigen::MatrixXd X_tmp(vb_, vb_);\n"
            "                    for (int P = 0; P < n_aux; ++P) {\n"
            "                        Eigen::Map<const Eigen::MatrixXd> B_P(B_ab_b.col(P).data(), vb_, vb_);\n"
            "                        X_tmp.noalias() = T_bb_pairs[k] * B_P;\n"
            "                        W_bb_pairs[k].noalias() += B_P.transpose() * X_tmp;\n"
            "                    }\n"
            "                }",
            "                contract_vvvv_pairs_df(B_ab_b, vb_, B_ab_b, vb_, n_aux, T_bb_pairs, W_bb_pairs);",
        ),
        (
            "            #pragma omp parallel for schedule(static)\n"
            "            for (int k = 0; k < n_pairs_ab; ++k) {\n"
            "                Eigen::MatrixXd X_tmp(va_, vb_);\n"
            "                for (int P = 0; P < n_aux; ++P) {\n"
            "                    Eigen::Map<const Eigen::MatrixXd> B_Pa(B_ab_a.col(P).data(), va_, va_);\n"
            "                    Eigen::Map<const Eigen::MatrixXd> B_Pb(B_ab_b.col(P).data(), vb_, vb_);\n"
            "                    X_tmp.noalias() = T_ab_pairs[k] * B_Pb;\n"
            "                    W_ab_pairs[k].noalias() += B_Pa.transpose() * X_tmp;\n"
            "                }\n"
            "            }",
            "            contract_vvvv_pairs_df(B_ab_a, va_, B_ab_b, vb_, n_aux, T_ab_pairs, W_ab_pairs);",
        ),
    ]

    for old_blk, new_blk in omp3_vvvv_blocks:
        if old_blk not in content:
            raise RuntimeError(f"Blok OMP3 vvvv tidak ditemukan:\n{old_blk[:100]}...")
        content = content.replace(old_blk, new_blk, 1)

    # 8. Deduplikasi lambda lokal di OMP3 -> gunakan helper global
    dup_lambda_re = re.compile(
        r"[ \t]*// Helper: Parallel Tiled DGEMM.*?\n"
        r"[ \t]*auto parallel_gemm_abt = \[\].*?\n"
        r"(?:.*\n)*?[ \t]*\};\n\n"
        r"[ \t]*// Helper: Zero-Copy 4D TBLIS View.*?\n"
        r"[ \t]*auto make_mat_view_4d = \[\].*?\n"
        r"(?:.*\n)*?[ \t]*\};\n\n",
        re.MULTILINE,
    )
    dup_lambda_plain_re = re.compile(
        r"[ \t]*auto parallel_gemm_abt = \[\].*?\n"
        r"(?:.*\n)*?[ \t]*\};\n\n"
        r"[ \t]*auto make_mat_view_4d = \[\].*?\n"
        r"(?:.*\n)*?[ \t]*\};\n\n",
        re.MULTILINE,
    )
    if dup_lambda_re.search(content):
        content = dup_lambda_re.sub("", content, count=1)
        if dup_lambda_plain_re.search(content):
            content = dup_lambda_plain_re.sub("", content, count=1)
        content = re.sub(r"\bparallel_gemm_abt\(", "parallel_gemm_abt_mp3(", content)
        content = re.sub(r"\bmake_mat_view_4d\(", "make_mat_view_4d_mp3(", content)

    return content, True


# =============================================================================
# RUNNER & BACKUP MANAGER
# =============================================================================
PATCHES = [
    ("Patch B (DF-ERI Schwarz & Tiled DGEMM)", Path("src/integrals/df_eri.cc"), patch_df_eri),
    ("Patch C (Direct SCF Thread-Local Raw Buffer)", Path("src/core/fock_builder.cc"), patch_fock_builder),
    ("Patch D1 (Symmetry Header Permutation Arrays)", Path("include/mshqc/symmetry/molecule_sym.h"), patch_molecule_sym_h),
    ("Patch D2 (Symmetry AO Permutation O(|G|N^2))", Path("src/symmetry/molecule_sym.cc"), patch_molecule_sym_cc),
    ("Patch E (MP3/OMP3 Zero-Copy & Ladder vvvv)", Path("src/mp3/mp3.cc"), patch_mp3),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject HPC patches B, C, D, E into mshqc.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Path root repositori mshqc")
    parser.add_argument("--dry-run", action="store_true", help="Validasi kecocokan anchor tanpa menulis file")
    parser.add_argument("--restore", action="store_true", help="Kembalikan seluruh file dari cadangan .bak")
    args = parser.parse_args()

    root: Path = args.root.resolve()

    if args.restore:
        restored = 0
        for label, rel_path, _ in PATCHES:
            target = root / rel_path
            bak = target.with_suffix(target.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, target)
                print(f"[RESTORED] {rel_path} <- {bak.name}")
                restored += 1
            else:
                print(f"[SKIP]     {rel_path} (tidak ada file .bak)")
        print(f"Selesai: {restored} file dipulihkan.")
        return 0

    modified_count = 0
    for label, rel_path, fn in PATCHES:
        target = root / rel_path
        if not target.exists():
            print(f"[ERROR]    File tidak ditemukan: {target}", file=sys.stderr)
            return 1

        original = target.read_text(encoding="utf-8")
        try:
            updated, changed = fn(original)
        except RuntimeError as exc:
            print(f"[FAIL]     {label} ({rel_path}):\n{exc}", file=sys.stderr)
            return 1

        if not changed:
            print(f"[OK/SKIP]  {label}: {rel_path} sudah ter-patch.")
            continue

        if args.dry_run:
            print(f"[DRY-RUN]  {label}: {rel_path} siap di-patch ({len(original)} -> {len(updated)} bytes).")
        else:
            bak = target.with_suffix(target.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(target, bak)
            target.write_text(updated, encoding="utf-8")
            print(f"[PATCHED]  {label}: {rel_path} (backup: {bak.name})")
        modified_count += 1

    if args.dry_run:
        print(f"\nDry-run selesai. {modified_count} file lolos validasi anchor.")
    else:
        print(f"\nSelesai: {modified_count} file diperbarui.")
    return 0


if __name__ == "__main__":
    sys.exit(main())