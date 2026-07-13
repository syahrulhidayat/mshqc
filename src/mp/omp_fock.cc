#include "mshqc/mp2/mp2.h"
#include "mshqc/integrals/eri_transformer.h" 
#include <omp.h>
#include <tblis/tblis.h>
#include <iostream> 

namespace mshqc {

using integrals::ERITransformer;

#define TBLIS_VIEW_4D(name, t, d1, d2, d3, d4) \
    tblis::varray_view<double> name({(tblis::len_type)d1, (tblis::len_type)d2, (tblis::len_type)d3, (tblis::len_type)d4}, t.data(), \
    {1, (tblis::stride_type)d1, (tblis::stride_type)(d1*d2), (tblis::stride_type)(d1*d2*d3)})

#define TBLIS_VIEW_3D(name, ptr, d1, d2, d3) \
    tblis::varray_view<double> name({(tblis::len_type)d1, (tblis::len_type)d2, (tblis::len_type)d3}, ptr, \
    {1, (tblis::stride_type)d1, (tblis::stride_type)(d1*d2)})

#define TBLIS_VIEW_2D(name, ptr, d1, d2) \
    tblis::varray_view<double> name({(tblis::len_type)d1, (tblis::len_type)d2}, ptr, \
    {1, (tblis::stride_type)d1})
void OMP2::build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b,
                           Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b)
{
    if (config_.eri_method != "exact") {
        Eigen::MatrixXd P_tot = P_a + P_b;
        int n_chol = scf_.L_mat.cols();

        Eigen::Map<const Eigen::MatrixXd> L_flat_J(scf_.L_mat.data(), nbf_ * nbf_, n_chol);
        Eigen::Map<const Eigen::VectorXd> P_tot_flat(P_tot.data(), nbf_ * nbf_);
        Eigen::VectorXd X_J = L_flat_J.transpose() * P_tot_flat; 
        Eigen::VectorXd J_flat = L_flat_J * X_J;                 
        Eigen::Map<Eigen::MatrixXd> J_mat(J_flat.data(), nbf_, nbf_);

        Eigen::MatrixXd Ka_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd Kb_mat = Eigen::MatrixXd::Zero(nbf_, nbf_);

        
            Eigen::MatrixXd Ka_priv = Eigen::MatrixXd::Zero(nbf_, nbf_);
            Eigen::MatrixXd Kb_priv = Eigen::MatrixXd::Zero(nbf_, nbf_);
            Eigen::MatrixXd Ta_buf(nbf_, nbf_), Tb_buf(nbf_, nbf_);

            
            for (int K = 0; K < n_chol; ++K) {
                Eigen::Map<const Eigen::MatrixXd> L_K(scf_.L_mat.col(K).data(), nbf_, nbf_);

                Ta_buf.noalias() = L_K * P_a;
                Ka_priv.noalias() += Ta_buf * L_K;

                if (nb_ > 0) {
                    Tb_buf.noalias() = L_K * P_b;
                    Kb_priv.noalias() += Tb_buf * L_K;
                }
            }
         
            { Ka_mat += Ka_priv; if (nb_ > 0) Kb_mat += Kb_priv; }
        

        F_a = H_core_ + J_mat - Ka_mat;
        if (nb_ > 0) F_b = H_core_ + J_mat - Kb_mat;
        else F_b = F_a;

        return;
    }

    Eigen::MatrixXd P_tot = P_a + P_b;
    double max_P = P_tot.cwiseAbs().maxCoeff(); 
    double threshold = 1e-9;
    const double* __restrict__ p_dtot = P_tot.data();
    const double* __restrict__ p_da   = P_a.data();
    const double* __restrict__ p_db   = P_b.data();

    const double* __restrict__ Jv = J_val_.data(); 
    const int* __restrict__ Ji = J_ind_.data(); 
    const size_t* __restrict__ Jp = J_ptr_.data();
    const double* __restrict__ Kv = K_val_.data(); 
    const int* __restrict__ Ki = K_ind_.data(); 
    const size_t* __restrict__ Kp = K_ptr_.data();

    int n_threads = omp_get_max_threads();
    Eigen::MatrixXd G_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd G_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    size_t n_rows = row_map_.size();

    #pragma omp parallel
    {
        Eigen::MatrixXd Ga_local = Eigen::MatrixXd::Zero(nbf_, nbf_);
        Eigen::MatrixXd Gb_local = Eigen::MatrixXd::Zero(nbf_, nbf_);

        #pragma omp for schedule(dynamic, 32)
        for (size_t r = 0; r < n_rows; ++r) {
            int mu = row_map_[r].first;
            int nu = row_map_[r].second;

            if (schwarz_(mu, nu) * max_P < threshold) continue;

            size_t js = Jp[r]; size_t je = Jp[r+1];
            size_t ks = Kp[r]; size_t ke = Kp[r+1];

            double vj = 0.0;
            #pragma omp simd reduction(+:vj)
            for (size_t k = js; k < je; ++k) vj += Jv[k] * p_dtot[Ji[k]];

            double ka = 0.0, kb = 0.0;
            #pragma omp simd reduction(+:ka, kb)
            for (size_t k = ks; k < ke; ++k) {
                double v = Kv[k]; int idx = Ki[k];
                ka += v * p_da[idx];
                kb += v * p_db[idx];
            }

            Ga_local(mu, nu) += vj - ka;
            Gb_local(mu, nu) += vj - kb;
        }

        #pragma omp critical
        {
            G_a += Ga_local;
            G_b += Gb_local;
        }
    }

    for (int i = 0; i < nbf_; ++i) {
        for (int j = 0; j < i; ++j) {
            G_a(j, i) = G_a(i, j); 
            G_b(j, i) = G_b(i, j);
        }
    }

    F_a = H_core_ + G_a;
    F_b = H_core_ + G_b;
}
void OMP2::build_opdm_alpha() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    if (is_restricted) { 
        auto* t_blk = t2_aa_.get_block(0,0,0,0);
        if (t_blk) {
            #pragma omp parallel for
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    double p_oo = 0.0;
                    for (int k = 0; k < na_; ++k) {
                        for (int a = 0; a < va_; ++a) {
                            for (int b = 0; b < va_; ++b) {
                                double t_ik = (*t_blk)(i, a, k, b);
                                double t_jk = (*t_blk)(j, a, k, b);
                                double t_jk_ex = (*t_blk)(j, b, k, a); 
                                p_oo += t_ik * (2.0 * t_jk - t_jk_ex);
                            }
                        }
                    }
                    G_oo_alpha_(i, j) = -2.0 * p_oo; 
                }
            }
            #pragma omp parallel for
            for (int a = 0; a < va_; ++a) {
                for (int b = 0; b < va_; ++b) {
                    double p_vv = 0.0;
                    for (int i = 0; i < na_; ++i) {
                        for (int j = 0; j < na_; ++j) {
                            for (int c = 0; c < va_; ++c) {
                                double t_ac = (*t_blk)(i, a, j, c);
                                double t_bc = (*t_blk)(i, b, j, c);
                                double t_cb = (*t_blk)(i, c, j, b);
                                p_vv += t_ac * (2.0 * t_bc - t_cb);
                            }
                        }
                    }
                    G_vv_alpha_(a, b) = 2.0 * p_vv; 
                }
            }
        }
        return; 
    }

    // --- UMP2 OPDM DENGAN EIGEN DGEMM (BEBAS OOM, ZERO-ALLOCATION) ---
    auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
    if (t_aa_blk) {
        // Blok G_oo_alpha_: Sudah optimal menggunakan Map.
        Eigen::Map<const Eigen::MatrixXd> T_mat(t_aa_blk->data(), na_, va_ * na_ * va_);
        G_oo_alpha_.noalias() = -0.5 * (T_mat * T_mat.transpose());

        // Blok G_vv_alpha_: PERBAIKAN FATAL OOM
        // Dimensi ColMajor (i, a, j, c) -> Stride i=1, a=na_, j=na_*va_, c=na_*va_*na_
        // Untuk j dan c yang tetap, elemen (i, a) membentuk matriks berurutan berukuran na_ x va_.
        const double* base_ptr = t_aa_blk->data();
        
        #pragma omp parallel
        {
            Eigen::MatrixXd G_vv_local = Eigen::MatrixXd::Zero(va_, va_);
            
            #pragma omp for schedule(dynamic)
            for (int jc = 0; jc < na_ * va_; ++jc) {
                int j = jc % na_;
                int c = jc / na_;
                
                size_t offset = j * (na_ * va_) + c * (na_ * va_ * na_);
                
                // M adalah pemetaan matriks T_ij (ukuran na_ x va_)
                Eigen::Map<const Eigen::MatrixXd> M(base_ptr + offset, na_, va_);
                
                // G_ab += (T_ij)^T * T_ij
                G_vv_local.noalias() += M.transpose() * M;
            }
            
            #pragma omp critical
            {
                G_vv_alpha_ += 0.5 * G_vv_local;
            }
        }
    }

    if (nb_ > 0 && vb_ > 0) {
        auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);
        if (t_ab_blk) {
            Eigen::Map<const Eigen::MatrixXd> Tab_mat(t_ab_blk->data(), na_, nb_ * va_ * vb_);
            G_oo_alpha_.noalias() -= Tab_mat * Tab_mat.transpose();

            const double* base_ab = t_ab_blk->data();
            
            #pragma omp parallel
            {
                Eigen::MatrixXd G_vv_local = Eigen::MatrixXd::Zero(va_, va_);
                
                #pragma omp for schedule(dynamic)
                for (int b = 0; b < vb_; ++b) {
                    size_t offset = b * (na_ * nb_ * va_);
                    
                    Eigen::Map<const Eigen::MatrixXd> M(base_ab + offset, na_ * nb_, va_);
                    G_vv_local.noalias() += M.transpose() * M;
                }
                
                #pragma omp critical
                {
                    G_vv_alpha_ += G_vv_local;
                }
            }
        }
    }
}

void OMP2::build_opdm_beta() {
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    if (nb_ == 0 || vb_ == 0) return;

    auto* t_bb_blk = t2_bb_.get_block(0,0,0,0);
    auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);

    if (t_bb_blk) {
        Eigen::Map<const Eigen::MatrixXd> T_mat(t_bb_blk->data(), nb_, vb_ * nb_ * vb_);
        G_oo_beta_.noalias() = -0.5 * (T_mat * T_mat.transpose());
        const double* base_bb = t_bb_blk->data();
        #pragma omp parallel
        {
            Eigen::MatrixXd G_vv_local = Eigen::MatrixXd::Zero(vb_, vb_);
            #pragma omp for schedule(dynamic)
            for (int jc = 0; jc < nb_ * vb_; ++jc) {
                int j = jc % nb_;
                int c = jc / nb_;
                
                size_t offset = j * (nb_ * vb_) + c * (nb_ * vb_ * nb_);
                Eigen::Map<const Eigen::MatrixXd> M(base_bb + offset, nb_, vb_);
                
                G_vv_local.noalias() += M.transpose() * M;
            }
            #pragma omp critical
            {
                G_vv_beta_ += 0.5 * G_vv_local;
            }
        }
    }

    if (t_ab_blk) {
        const double* base_ab = t_ab_blk->data();
        #pragma omp parallel
        {
            Eigen::MatrixXd G_oo_local = Eigen::MatrixXd::Zero(nb_, nb_);
            #pragma omp for schedule(dynamic)
            for (int ab = 0; ab < va_ * vb_; ++ab) {
                int a = ab % va_;
                int b = ab / va_;
                
                size_t offset = a * (na_ * nb_) + b * (na_ * nb_ * va_);
                Eigen::Map<const Eigen::MatrixXd> M(base_ab + offset, na_, nb_);
                
                G_oo_local.noalias() += M.transpose() * M;
            }
            #pragma omp critical
            {
                G_oo_beta_ -= G_oo_local;
            }
        }

        Eigen::Map<const Eigen::MatrixXd> M_vv(base_ab, na_ * nb_ * va_, vb_);
        G_vv_beta_.noalias() += M_vv.transpose() * M_vv;
    }
}
void OMP2::build_generalized_fock() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_) = G_oo_alpha_; 
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();

    Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (is_restricted) {
        G_oo_beta_ = G_oo_alpha_;
        G_vv_beta_ = G_vv_alpha_;
        G_full_b = G_full_a;
        P_corr_b = P_corr_a;
    } else if (nb_ > 0) {
        G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    }

    Eigen::MatrixXd F_HF_ao_a, F_HF_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_HF_ao_a, F_HF_ao_b);
    Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;

    Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;

    Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
    build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
    G_gamma_ao_a -= H_core_;
    if (nb_ > 0) G_gamma_ao_b -= H_core_;

    Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
    Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (nb_ > 0) G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;

    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

    if (config_.eri_method == "exact") {
        Z_mat_a.setZero();
        Z_mat_b.setZero();
        if (is_restricted) {
            if (config_.print_level > 0) std::cout << "  [DEBUG] Memulai evaluasi Z-Vector RMP2 Exact (Analitik Spasial)..." << std::endl;
            const auto& eri_ao = integrals_->compute_eri();
            const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
            const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
            
            auto ovvv = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Ca_v, Ca_v, nbf_, na_, va_, va_, va_);
            auto ooov = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_o, Ca_o, Ca_v, nbf_, na_, na_, na_, va_);
            auto* t_blk = t2_aa_.get_block(0,0,0,0);
            
            if (t_blk) {
                #pragma omp parallel for
                for (int a=0; a<va_; ++a) {
                    for (int i=0; i<na_; ++i) {
                        double z1 = 0.0, z2 = 0.0;
                        for (int j=0; j<na_; ++j) {
                            for (int c=0; c<va_; ++c) {
                                for (int b=0; b<va_; ++b) {
                                    // PERBAIKAN: Amplitudo spasial T(i,b,j,c) dan T(i,c,j,b)
                                    double tau = 2.0 * (*t_blk)(i, b, j, c) - (*t_blk)(i, c, j, b);
                                    z1 += tau * ovvv(j, c, a, b);
                                }
                                for (int k=0; k<na_; ++k) {
                                    // PERBAIKAN: Amplitudo spasial T(j,a,k,c) dan T(j,c,k,a)
                                    double tau = 2.0 * (*t_blk)(j, a, k, c) - (*t_blk)(j, c, k, a);
                                    z2 += tau * ooov(j, i, k, c);
                                }
                            }
                        }
                        // PERBAIKAN: Kalikan dengan 4.0 untuk total spin Lagrangian
                        Z_mat_a(a, i) = 4.0 * (z1 - z2);
                    }
                }
            }
            Z_mat_b = Z_mat_a;
        } else {
            if (scf_.irreps_alpha.empty()) scf_.irreps_alpha.assign(nbf_, 0);
            if (scf_.irreps_beta.empty()) scf_.irreps_beta.assign(nbf_, 0);

            if (config_.print_level > 0) std::cout << "  [DEBUG] Memulai evaluasi Z-Vector O(N^5) MO-Driven UMP2 (TBLIS)..." << std::endl;

            const auto& eri_ao = integrals_->compute_eri();
            const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
            const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);

            auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
            auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

            auto ovvv_blk = integrals::ERITransformer::transform_ovvv_blocked(eri_ao, Ca_o, Ca_v, occ_spaces_a, vir_spaces_a, nbf_);
            auto ooov_blk = integrals::ERITransformer::transform_ooov_blocked(eri_ao, Ca_o, Ca_v, occ_spaces_a, vir_spaces_a, nbf_);

            BlockedTensor4D* T2_ptr = &t2_aa_;

            #pragma omp parallel
            {
                Eigen::MatrixXd Z_local = Eigen::MatrixXd::Zero(va_, na_);
                std::vector<double> z_buffer(na_ * va_, 0.0);

                #pragma omp for schedule(dynamic)
                for (int s_i = 0; s_i < occ_spaces_a.size(); ++s_i) {
                    const auto& o_i = occ_spaces_a[s_i];
                    if (o_i.size == 0) continue; 

                    for (const auto& v_a : vir_spaces_a) {
                        if (v_a.size == 0) continue; 
                        if ((o_i.id ^ v_a.id) != 0) continue; 

                        for (const auto& o_j : occ_spaces_a) {
                            if (o_j.size == 0) continue; 

                            for (const auto& v_b : vir_spaces_a) {
                                if (v_b.size == 0) continue; 

                                for (const auto& v_c : vir_spaces_a) {
                                    if (v_c.size == 0) continue; 

                                    if ((o_j.id ^ v_c.id ^ v_a.id ^ v_b.id) == 0) {
                                        auto* blk = ovvv_blk.get_block(o_j.id, v_c.id, v_a.id, v_b.id);
                                        auto* t_blk = T2_ptr->get_block(o_i.id, v_b.id, o_j.id, v_c.id);

                                        if (blk && t_blk) {
                                            tblis::tblis_tensor t_T, t_V, t_Z;

                                            tblis::len_type ni = o_i.size, na = v_a.size, nb = v_b.size, nj = o_j.size, nc = v_c.size;

                                            tblis::len_type len_T[] = {ni, nb, nj, nc};
                                            tblis::stride_type str_T[] = {1, ni, ni*nb, ni*nb*nj};
                                            tblis::tblis_init_tensor_d(&t_T, 4, len_T, t_blk->data(), str_T);

                                            tblis::len_type len_V[] = {nj, nc, na, nb};
                                            tblis::stride_type str_V[] = {1, nj, nj*nc, nj*nc*na};
                                            tblis::tblis_init_tensor_d(&t_V, 4, len_V, blk->data(), str_V);

                                            Eigen::Map<Eigen::MatrixXd> Z_temp(z_buffer.data(), ni, na);
                                            Z_temp.setZero(); 

                                            tblis::len_type len_Z[] = {ni, na};
                                            tblis::stride_type str_Z[] = {1, ni};
                                            tblis::tblis_init_tensor_d(&t_Z, 2, len_Z, Z_temp.data(), str_Z);

                                            tblis::tblis_tensor_mult(nullptr, nullptr, &t_T, "ibjc", &t_V, "jcab", &t_Z, "ia");

                                            for(int di=0; di<ni; ++di) {
                                                for(int da=0; da<na; ++da) {
                                                    Z_local(v_a.offset + da, o_i.offset + di) += Z_temp(di, da);
                                                }
                                            }
                                        }
                                    }
                                }

                                for (const auto& o_k : occ_spaces_a) {
                                    if (o_k.size == 0) continue; 

                                    if ((o_i.id ^ o_j.id ^ o_k.id ^ v_b.id) == 0) {
                                        auto* blk = ooov_blk.get_block(o_j.id, o_i.id, o_k.id, v_b.id);
                                        auto* t_blk = T2_ptr->get_block(o_j.id, v_a.id, o_k.id, v_b.id);

                                        if (blk && t_blk) {
                                            tblis::tblis_tensor t_V, t_T, t_Z;
                                            tblis::len_type ni = o_i.size, na = v_a.size, nj = o_j.size, nk = o_k.size, nb = v_b.size;

                                            tblis::len_type len_V[] = {nj, ni, nk, nb};
                                            tblis::stride_type str_V[] = {1, nj, nj*ni, nj*ni*nk};
                                            tblis::tblis_init_tensor_d(&t_V, 4, len_V, blk->data(), str_V);

                                            tblis::len_type len_T[] = {nj, na, nk, nb};
                                            tblis::stride_type str_T[] = {1, nj, nj*na, nj*na*nk};
                                            tblis::tblis_init_tensor_d(&t_T, 4, len_T, t_blk->data(), str_T);

                                            Eigen::Map<Eigen::MatrixXd> Z_temp(z_buffer.data(), ni, na);
                                            Z_temp.setZero(); 

                                            tblis::len_type len_Z[] = {ni, na};
                                            tblis::stride_type str_Z[] = {1, ni};
                                            tblis::tblis_init_tensor_d(&t_Z, 2, len_Z, Z_temp.data(), str_Z);

                                            tblis::tblis_tensor_mult(nullptr, nullptr, &t_V, "jikb", &t_T, "jakb", &t_Z, "ia");

                                            for(int di=0; di<ni; ++di) {
                                                for(int da=0; da<na; ++da) {
                                                    Z_local(v_a.offset + da, o_i.offset + di) -= Z_temp(di, da); 
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp critical
                Z_mat_a += Z_local;
            }

            if (nb_ > 0 && vb_ > 0 && !t2_bb_.blocks.empty() && !t2_ab_.blocks.empty()) {
                const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
                const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);
                auto ovvv_bb = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Cb_v, Cb_v, nbf_, nb_, vb_, vb_, vb_);
                auto ooov_bb = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_o, Cb_o, Cb_v, nbf_, nb_, nb_, nb_, vb_);
                auto ovvv_ba_aa = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_v, Ca_v, Ca_v, nbf_, nb_, vb_, va_, va_);
                auto ooov_aa_bb = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_o, Cb_o, Cb_v, nbf_, na_, na_, nb_, vb_);
                auto ovvv_ab_bb = integrals::ERITransformer::transform_custom(eri_ao, Ca_o, Ca_v, Cb_v, Cb_v, nbf_, na_, va_, vb_, vb_);
                auto ooov_bb_aa = integrals::ERITransformer::transform_custom(eri_ao, Cb_o, Cb_o, Ca_o, Ca_v, nbf_, nb_, nb_, na_, va_);

                auto* t_bb_dense = t2_bb_.get_block(0,0,0,0);
                auto* t_ab_dense = t2_ab_.get_block(0,0,0,0);

                if (t_bb_dense && t_ab_dense) {
                    #pragma omp parallel for
                    for (int i = 0; i < nb_; ++i) {
                        for (int a = 0; a < vb_; ++a) {
                            double z1 = 0.0, z2 = 0.0;
                            for (int j = 0; j < nb_; ++j) {
                                for (int b = 0; b < vb_; ++b) {
                                    for (int c = 0; c < vb_; ++c) z1 += (*t_bb_dense)(i, j, b, c) * ovvv_bb(j, c, a, b);
                                    for (int k = 0; k < nb_; ++k) z2 += (*t_bb_dense)(j, k, a, b) * ooov_bb(j, i, k, b);
                                }
                            }
                            Z_mat_b(a, i) += z1 - z2;
                        }
                    }

                    #pragma omp parallel for
                    for (int i = 0; i < na_; ++i) {
                        for (int a = 0; a < va_; ++a) {
                            double z1 = 0.0, z2 = 0.0;
                            for (int j = 0; j < nb_; ++j) {
                                for (int b = 0; b < va_; ++b) {
                                    for (int c = 0; c < vb_; ++c) z1 += (*t_ab_dense)(i, j, b, c) * ovvv_ba_aa(j, c, a, b);
                                }
                            }
                            for (int j = 0; j < na_; ++j) {
                                for (int b = 0; b < vb_; ++b) {
                                    for (int k = 0; k < nb_; ++k) z2 += (*t_ab_dense)(j, k, a, b) * ooov_aa_bb(j, i, k, b);
                                }
                            }
                            Z_mat_a(a, i) += z1 - z2;
                        }
                    }

                    #pragma omp parallel for
                    for (int i = 0; i < nb_; ++i) {
                        for (int a = 0; a < vb_; ++a) {
                            double z1 = 0.0, z2 = 0.0;
                            for (int j = 0; j < na_; ++j) {
                                for (int b = 0; b < vb_; ++b) {
                                    for (int c = 0; c < va_; ++c) z1 += (*t_ab_dense)(j, i, c, b) * ovvv_ab_bb(j, c, a, b);
                                }
                            }
                            for (int j = 0; j < nb_; ++j) {
                                for (int b = 0; b < va_; ++b) {
                                    for (int k = 0; k < na_; ++k) z2 += (*t_ab_dense)(k, j, b, a) * ooov_bb_aa(j, i, k, b);
                                }
                            }
                            Z_mat_b(a, i) += z1 - z2;
                        }
                    }
                }
            }
            if (config_.print_level > 0) std::cout << "  [DEBUG] Evaluasi Z-Vector UMP2 Selesai!" << std::endl;
        }
    } else if (config_.eri_method == "cholesky") {
        evaluate_z_vector_cholesky(Z_mat_a, Z_mat_b);
        if (is_restricted) Z_mat_b = Z_mat_a;
    } else {
        if (config_.print_level > 0) std::cout << "  [DEBUG] Memulai evaluasi Z-Vector O(N^4) Density Fitting..." << std::endl;

        int n_aux = scf_.L_mat.cols();

        Eigen::MatrixXd T2_aa = Eigen::MatrixXd::Zero(na_*va_, na_*va_);
        auto* t_aa_blk = t2_aa_.get_block(0,0,0,0);
        if (t_aa_blk) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < na_; ++i) {
                for (int j = 0; j < na_; ++j) {
                    for (int a = 0; a < va_; ++a) {
                        for (int b = 0; b < va_; ++b) {
                            double L_ijab = 0.0;
                            if (is_restricted) {
                                L_ijab = 8.0 * (*t_aa_blk)(i, a, j, b) - 4.0 * (*t_aa_blk)(i, b, j, a);
                            } else {
                                L_ijab = (*t_aa_blk)(i, a, j, b);
                            }
                            T2_aa(i*va_+a, j*va_+b) = L_ijab; 
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd X_a = T2_aa * B_ia_P_alpha_;
        
        Eigen::MatrixXd B_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
        Eigen::MatrixXd B_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);

        const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
        const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);

        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
            Eigen::MatrixXd MO_oo_a = Ca_o.transpose() * (B_AO * Ca_o);
            Eigen::MatrixXd MO_vv_a = Ca_v.transpose() * (B_AO * Ca_v);

            for(int i=0; i<na_; ++i) for(int j=0; j<na_; ++j) B_oo_a(i*na_+j, P) = MO_oo_a(i, j);
            for(int a=0; a<va_; ++a) for(int b=0; b<va_; ++b) B_vv_a(a*va_+b, P) = MO_vv_a(a, b);
        }

        Z_mat_a.setZero();

        for (int P = 0; P < n_aux; ++P) {
        
            Eigen::Map<const Eigen::MatrixXd> XT_a(X_a.col(P).data(), va_, na_); 
            Eigen::Map<const Eigen::MatrixXd> V_a(B_vv_a.col(P).data(), va_, va_);
            Eigen::Map<const Eigen::MatrixXd> O_a(B_oo_a.col(P).data(), na_, na_);
            
            
            Z_mat_a.noalias() += V_a * XT_a - XT_a * O_a;
        }

        if (is_restricted) {
            Z_mat_b = Z_mat_a;
        } else if (nb_ > 0 && vb_ > 0) {
            Eigen::MatrixXd T2_ab = Eigen::MatrixXd::Zero(na_*va_, nb_*vb_);
            Eigen::MatrixXd T2_bb = Eigen::MatrixXd::Zero(nb_*vb_, nb_*vb_);
            
            auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);
            if (t_ab_blk) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < na_; ++i) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int a = 0; a < va_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                T2_ab(i*va_+a, j*vb_+b) = (*t_ab_blk)(i, j, a, b);
                            }
                        }
                    }
                }
            }
            auto* t_bb_blk = t2_bb_.get_block(0,0,0,0);
            if (t_bb_blk) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nb_; ++i) {
                    for (int j = 0; j < nb_; ++j) {
                        for (int a = 0; a < vb_; ++a) {
                            for (int b = 0; b < vb_; ++b) {
                                T2_bb(i*vb_+a, j*vb_+b) = (*t_bb_blk)(i, j, a, b);
                            }
                        }
                    }
                }
            }
            
            X_a += T2_ab * B_ia_P_beta_;
            Eigen::MatrixXd X_b = T2_bb * B_ia_P_beta_ + T2_ab.transpose() * B_ia_P_alpha_;

            Eigen::MatrixXd B_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux);
            Eigen::MatrixXd B_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux);
            const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
            const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

            for (int P = 0; P < n_aux; ++P) {
                Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
                Eigen::MatrixXd MO_oo_b = Cb_o.transpose() * (B_AO * Cb_o);
                Eigen::MatrixXd MO_vv_b = Cb_v.transpose() * (B_AO * Cb_v);
                for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) B_oo_b(i*nb_+j, P) = MO_oo_b(i, j);
                for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b) B_vv_b(a*vb_+b, P) = MO_vv_b(a, b);
            }

            Z_mat_b.setZero();
           
            for (int P = 0; P < n_aux; ++P) {
                Eigen::Map<const Eigen::MatrixXd> XT_b(X_b.col(P).data(), vb_, nb_); 
                Eigen::Map<const Eigen::MatrixXd> V_b(B_vv_b.col(P).data(), vb_, vb_);
                Eigen::Map<const Eigen::MatrixXd> O_b(B_oo_b.col(P).data(), nb_, nb_);
                Z_mat_b.noalias() += V_b * XT_b - XT_b * O_b;
            }
        }
    }
    F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
    if (na_ > 0 && va_ > 0) {
        Eigen::MatrixXd F_vo_a = F_gen_a_.block(na_, 0, va_, na_);
        Eigen::MatrixXd L_sep_a = G_vv_alpha_ * F_vo_a - F_vo_a * G_oo_alpha_;

        F_gen_a_.block(na_, 0, va_, na_) += L_sep_a;
        F_gen_a_.block(0, na_, na_, va_) += L_sep_a.transpose();

        F_gen_a_.block(na_, 0, va_, na_) += Z_mat_a;
        F_gen_a_.block(0, na_, na_, va_) += Z_mat_a.transpose();
    }

    F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
    if (nb_ > 0 && vb_ > 0) {
        Eigen::MatrixXd F_vo_b = F_gen_b_.block(nb_, 0, vb_, nb_);
        Eigen::MatrixXd L_sep_b = G_vv_beta_ * F_vo_b - F_vo_b * G_oo_beta_;

        F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += L_sep_b.transpose();

        F_gen_b_.block(nb_, 0, vb_, nb_) += Z_mat_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += Z_mat_b.transpose();
    }
}
}