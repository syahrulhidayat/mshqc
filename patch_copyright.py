import os
import sys

def restore_pure_omp3(filepath="src/mp3/mp3.cc"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan!")
        sys.exit(1)

    with open(filepath, 'r') as file:
        content = file.read()

    start_str = "void OMP3::build_opdm_alpha() {"
    end_str = "void OMP3::build_hessian_diagonal(Eigen::VectorXd& diag_H, double grad_norm) {"
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)

    if start_idx == -1 or end_idx == -1:
        print("[ERROR] Batas fungsi tidak ditemukan!")
        sys.exit(1)

    # Restorasi kode ASLI (FDF O(N^5)) yang 100% valid secara matematis,
    # dengan perbaikan L_sep dan pencegahan segfault index
    restored_code = """void OMP3::build_opdm_alpha() {
    if (L2_aa_.size() == 0) { OMP2::build_opdm_alpha(); return; }

    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    if(!t2_aa_dense) return;

    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_alpha_ = Eigen::MatrixXd::Zero(na_, na_);
    G_vv_alpha_ = Eigen::MatrixXd::Zero(va_, va_);
    
    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i=0; i<na_; ++i)
        for(int j=0; j<na_; ++j)
            for(int a=0; a<va_; ++a)
                for(int b=0; b<va_; ++b)
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
                    
    if (is_restricted) {
        Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);
        Eigen::Tensor<double, 4> L2_tilde(na_, na_, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i=0; i<na_; ++i) {
            for (int j=0; j<na_; ++j) {
                for (int a=0; a<va_; ++a) {
                    for (int b=0; b<va_; ++b) {
                        T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);
                        L2_tilde(i,j,a,b) = 2.0 * L2_aa_(i,j,a,b) - L2_aa_(i,j,b,a);
                    }
                }
            }
        }
        
        TBLIS_VIEW_4D(t_T2, T2_aa_ijab, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_L2, L2_aa_, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_L2t, L2_tilde, na_, na_, va_, va_);
        TBLIS_VIEW_2D(t_Goo, G_oo_alpha_.data(), na_, na_);
        TBLIS_VIEW_2D(t_Gvv, G_vv_alpha_.data(), va_, va_);

        tblis::mult<double>(-1.0, t_T2, "ikab", t_T2t, "jkab", 0.0, t_Goo, "ij");
        tblis::mult<double>(-0.5, t_T2, "ikab", t_L2t, "jkab", 1.0, t_Goo, "ij"); 
        tblis::mult<double>(-0.5, t_L2, "ikab", t_T2t, "jkab", 1.0, t_Goo, "ij");  

        tblis::mult<double>(1.0, t_T2, "ijac", t_T2t, "ijbc", 0.0, t_Gvv, "ab");
        tblis::mult<double>(0.5, t_T2, "ijac", t_L2t, "ijbc", 1.0, t_Gvv, "ab");
        tblis::mult<double>(0.5, t_L2, "ijac", t_T2t, "ijbc", 1.0, t_Gvv, "ab");
        return;
    }

    TBLIS_VIEW_4D(t_T2aa, T2_aa_ijab, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T3aa, L2_aa_, na_, na_, va_, va_);
    TBLIS_VIEW_2D(t_Goo_a, G_oo_alpha_.data(), na_, na_);
    TBLIS_VIEW_2D(t_Gvv_a, G_vv_alpha_.data(), va_, va_);

    tblis::mult<double>(-0.5, t_T2aa, "ikab", t_T2aa, "jkab", 1.0, t_Goo_a, "ij");
    tblis::mult<double>(-0.5, t_T2aa, "ikab", t_T3aa, "jkab", 1.0, t_Goo_a, "ij"); 
    tblis::mult<double>(-0.5, t_T3aa, "ikab", t_T2aa, "jkab", 1.0, t_Goo_a, "ij"); 

    tblis::mult<double>(0.5, t_T2aa, "ijac", t_T2aa, "ijbc", 1.0, t_Gvv_a, "ab");
    tblis::mult<double>(0.5, t_T2aa, "ijac", t_T3aa, "ijbc", 1.0, t_Gvv_a, "ab");  
    tblis::mult<double>(0.5, t_T3aa, "ijac", t_T2aa, "ijbc", 1.0, t_Gvv_a, "ab");

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
        TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
        TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);
        
        tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij");
        tblis::mult<double>(-0.5, t_T2ab, "ikab", t_T3ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(-0.5, t_T3ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a, "ij"); 
        tblis::mult<double>(1.0, t_T2ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab");
        tblis::mult<double>(0.5, t_T2ab, "ijac", t_T3ab, "ijbc", 1.0, t_Gvv_a, "ab");  
        tblis::mult<double>(0.5, t_T3ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a, "ab");
    }
}

void OMP3::build_opdm_beta() {
    if (L2_bb_.size() == 0 && L2_ab_.size() == 0) { OMP2::build_opdm_beta(); return; }
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
    G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
    
    if (is_restricted || nb_ == 0 || vb_ == 0) return; 
    auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
    Eigen::Tensor<double, 4> dummy_bb;
    if (!t2_bb_dense && nb_ > 0 && vb_ > 0) {
        dummy_bb = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_);
        dummy_bb.setZero();
        t2_bb_dense = &dummy_bb;
    }
    auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
    Eigen::Tensor<double, 4> dummy_ab;
    if (!t2_ab_dense && nb_ > 0 && vb_ > 0) {
        dummy_ab = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_);
        dummy_ab.setZero();
        t2_ab_dense = &dummy_ab;
    }

    TBLIS_VIEW_4D(t_T2bb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
    TBLIS_VIEW_4D(t_T3bb, L2_bb_, nb_, nb_, vb_, vb_);
    TBLIS_VIEW_4D(t_T2ab, (*t2_ab_dense), na_, nb_, va_, vb_);
    TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);
    TBLIS_VIEW_2D(t_Goo_b, G_oo_beta_.data(), nb_, nb_);
    TBLIS_VIEW_2D(t_Gvv_b, G_vv_beta_.data(), vb_, vb_);

    tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij");
    tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T3bb, "jkab", 1.0, t_Goo_b, "ij"); 
    tblis::mult<double>(-0.5, t_T3bb, "ikab", t_T2bb, "jkab", 1.0, t_Goo_b, "ij"); 

    tblis::mult<double>(0.5, t_T2bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab");
    tblis::mult<double>(0.5, t_T2bb, "ijac", t_T3bb, "ijbc", 1.0, t_Gvv_b, "ab");  
    tblis::mult<double>(0.5, t_T3bb, "ijac", t_T2bb, "ijbc", 1.0, t_Gvv_b, "ab"); 
    
    tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij");
    tblis::mult<double>(-0.5, t_T2ab, "kiab", t_T3ab, "kjab", 1.0, t_Goo_b, "ij");  
    tblis::mult<double>(-0.5, t_T3ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b, "ij");  
    
    tblis::mult<double>(1.0, t_T2ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab");
    tblis::mult<double>(0.5, t_T2ab, "ijca", t_T3ab, "ijcb", 1.0, t_Gvv_b, "ab");   
    tblis::mult<double>(0.5, t_T3ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b, "ab");
}

void OMP3::build_generalized_fock() {
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
    G_full_a.block(0, 0, na_, na_) = G_oo_alpha_; 
    G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
    Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();
    
    Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    
    if (!is_restricted && nb_ > 0) {
        G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
        G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
        P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
    } else if (is_restricted) { P_corr_b = P_corr_a; }

    Eigen::MatrixXd F_HF_ao_a, F_HF_ao_b;
    build_fock_fast(scf_.P_alpha, scf_.P_beta, F_HF_ao_a, F_HF_ao_b);
    
    Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;
    Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (!is_restricted && nb_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;
    else if (is_restricted) F_HF_mo_b = F_HF_mo_a;

    Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
    build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
    
    G_gamma_ao_a -= H_core_;
    if (!is_restricted && nb_ > 0) G_gamma_ao_b -= H_core_;
    else if (is_restricted) G_gamma_ao_b = G_gamma_ao_a;
    
    Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
    Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
    if (!is_restricted && nb_ > 0) G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;
    else if (is_restricted) G_gamma_mo_b = G_gamma_mo_a;

    F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
    if (na_ > 0 && va_ > 0) {
        Eigen::MatrixXd F_HF_vo_a = F_HF_mo_a.block(na_, 0, va_, na_);
        // KOREKSI KUNCI: F_vo * G_oo - G_vv * F_vo
        Eigen::MatrixXd L_sep_a = F_HF_vo_a * G_oo_alpha_ - G_vv_alpha_ * F_HF_vo_a;
        F_gen_a_.block(na_, 0, va_, na_) += L_sep_a;
        F_gen_a_.block(0, na_, na_, va_) += L_sep_a.transpose();
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
        Eigen::MatrixXd F_HF_vo_b = F_HF_mo_b.block(nb_, 0, vb_, nb_);
        // KOREKSI KUNCI: F_vo * G_oo - G_vv * F_vo
        Eigen::MatrixXd L_sep_b = F_HF_vo_b * G_oo_beta_ - G_vv_beta_ * F_HF_vo_b;
        F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += L_sep_b.transpose();
    } else if (is_restricted) { F_gen_b_ = F_gen_a_; }

    int n_aux = scf_.L_mat.cols();
    Eigen::MatrixXd B_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
    Eigen::MatrixXd B_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);
    Eigen::MatrixXd B_oo_b, B_vv_b;
    
    if (!is_restricted && nb_ > 0 && vb_ > 0) { 
        B_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux); 
        B_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux); 
    }
    
    const Eigen::MatrixXd& Ca_o = scf_.C_alpha.leftCols(na_);
    const Eigen::MatrixXd& Ca_v = scf_.C_alpha.rightCols(va_);
    const Eigen::MatrixXd& Cb_o = scf_.C_beta.leftCols(nb_);
    const Eigen::MatrixXd& Cb_v = scf_.C_beta.rightCols(vb_);

    #pragma omp parallel
    {
        Eigen::MatrixXd priv_oo_a = Eigen::MatrixXd::Zero(na_*na_, n_aux);
        Eigen::MatrixXd priv_vv_a = Eigen::MatrixXd::Zero(va_*va_, n_aux);
        Eigen::MatrixXd priv_oo_b, priv_vv_b;
        if (!is_restricted && nb_ > 0) {
            priv_oo_b = Eigen::MatrixXd::Zero(nb_*nb_, n_aux);
            priv_vv_b = Eigen::MatrixXd::Zero(vb_*vb_, n_aux);
        }
        
        #pragma omp for schedule(dynamic)
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> B_AO(scf_.L_mat.col(P).data(), nbf_, nbf_);
            Eigen::MatrixXd MO_oo_a = Ca_o.transpose() * (B_AO * Ca_o);
            Eigen::MatrixXd MO_vv_a = Ca_v.transpose() * (B_AO * Ca_v);
            for(int i=0; i<na_; ++i) for(int j=0; j<na_; ++j) priv_oo_a(i*na_+j, P) = MO_oo_a(i, j);
            for(int a=0; a<va_; ++a) for(int b=0; b<va_; ++b) priv_vv_a(a*va_+b, P) = MO_vv_a(a, b);
            
            if (!is_restricted && nb_ > 0 && vb_ > 0) {
                Eigen::MatrixXd MO_oo_b = Cb_o.transpose() * (B_AO * Cb_o);
                Eigen::MatrixXd MO_vv_b = Cb_v.transpose() * (B_AO * Cb_v);
                for(int i=0; i<nb_; ++i) for(int j=0; j<nb_; ++j) priv_oo_b(i*nb_+j, P) = MO_oo_b(i, j);
                for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b) priv_vv_b(a*vb_+b, P) = MO_vv_b(a, b);
            }
        }
        #pragma omp critical
        {
            B_oo_a += priv_oo_a; B_vv_a += priv_vv_a;
            if (!is_restricted && nb_ > 0) { B_oo_b += priv_oo_b; B_vv_b += priv_vv_b; }
        }
    }

    auto* t2_aa_dense = t2_aa_.get_block(0,0,0,0);
    Eigen::Tensor<double, 4> T2_aa_ijab(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for(int i=0; i<na_; ++i)
        for(int j=0; j<na_; ++j)
            for(int a=0; a<va_; ++a)
                for(int b=0; b<va_; ++b)
                    T2_aa_ijab(i,j,a,b) = (*t2_aa_dense)(i,a,j,b);
    TBLIS_VIEW_4D(t_Taa, T2_aa_ijab, na_, na_, va_, va_);
    
    Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);
    Eigen::MatrixXd Z_mat_b;
    if (!is_restricted && nb_ > 0 && vb_ > 0) Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

    TBLIS_VIEW_3D(t_Bvv_a, B_vv_a.data(), va_, va_, n_aux);
    TBLIS_VIEW_3D(t_Boo_a, B_oo_a.data(), na_, na_, n_aux);
    TBLIS_VIEW_2D(t_Za, Z_mat_a.data(), va_, na_);
    TBLIS_VIEW_3D(t_Bia_a, B_ia_P_alpha_.data(), va_, na_, n_aux); 

    // =========================================================================
    // RESTRICTED FDF EXACT RESTORATION
    // =========================================================================
    if (is_restricted) {
        Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for(int i=0; i<na_; ++i) {
            for(int j=0; j<na_; ++j) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<va_; ++b) {
                        T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);
                    }
                }
            }
        }
        TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);

        // OVOV
        Eigen::Tensor<double, 4> Gamma_ovov_aa(na_, va_, na_, va_); Gamma_ovov_aa.setZero();
        TBLIS_VIEW_4D(t_Govov_aa, Gamma_ovov_aa, na_, va_, na_, va_);
        tblis::mult<double>(-0.5, t_T2t, "imae", t_Taa, "jmeb", 1.0, t_Govov_aa, "iajb"); 
        tblis::mult<double>(-0.5, t_T2t, "mjea", t_Taa, "mibe", 1.0, t_Govov_aa, "iajb"); 
        tblis::mult<double>(0.25, t_T2t, "mjea", t_Taa, "miba", 1.0, t_Govov_aa, "iajb"); 
        tblis::mult<double>(0.25, t_T2t, "imae", t_Taa, "jmba", 1.0, t_Govov_aa, "iajb"); 
        
        Eigen::MatrixXd G_mat_aa(na_ * va_, na_ * va_);
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=0; i<na_; ++i)
            for(int a=0; a<va_; ++a)
                for(int j=0; j<na_; ++j)
                    for(int b=0; b<va_; ++b)
                        G_mat_aa(i * va_ + a, j * va_ + b) = Gamma_ovov_aa(i, a, j, b);
        
        Eigen::MatrixXd Y_aa = G_mat_aa * B_ia_P_alpha_;
        TBLIS_VIEW_3D(t_Y_aa, Y_aa.data(), va_, na_, n_aux);
        tblis::mult<double>(-2.0, t_Y_aa, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");
        tblis::mult<double>( 2.0, t_Y_aa, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");

        // VVVV O(N^5) FDF
        Eigen::Tensor<double, 3> X_ij_P_a(na_, na_, n_aux);
        TBLIS_VIEW_3D(t_Wij_a, X_ij_P_a.data(), na_, na_, n_aux);
        tblis::mult<double>(1.0, t_Taa, "ijcd", t_Bvv_a, "cdP", 0.0, t_Wij_a, "ijP");

        Eigen::Tensor<double, 3> Y_ab_P_a(va_, va_, n_aux);
        TBLIS_VIEW_3D(t_Xab_a, Y_ab_P_a.data(), va_, va_, n_aux);
        tblis::mult<double>(0.25, t_T2t, "ijab", t_Wij_a, "ijP", 0.0, t_Xab_a, "abP");
        tblis::mult<double>(4.0, t_Xab_a, "abP", t_Bia_a, "biP", 1.0, t_Za, "ai");

        // OOOO O(N^5) FDF
        Eigen::Tensor<double, 3> W_ab_P_a(va_, va_, n_aux);
        TBLIS_VIEW_3D(t_Wab_a, W_ab_P_a.data(), va_, va_, n_aux);
        tblis::mult<double>(1.0, t_Taa, "klab", t_Boo_a, "klP", 0.0, t_Wab_a, "abP");

        Eigen::Tensor<double, 3> X_ij_P_a_oo(na_, na_, n_aux);
        TBLIS_VIEW_3D(t_Xij_a, X_ij_P_a_oo.data(), na_, na_, n_aux);
        tblis::mult<double>(0.25, t_T2t, "ijab", t_Wab_a, "abP", 0.0, t_Xij_a, "ijP");
        tblis::mult<double>(-4.0, t_Xij_a, "ijP", t_Bia_a, "ajP", 1.0, t_Za, "ai");

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Teff_aa(na_*va_, na_*va_);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < na_; ++j) {
                    for (int b = 0; b < va_; ++b) {
                        double t1_dir = T2_aa_ijab(i, j, a, b);
                        double t2_dir = L2_aa_(i, j, a, b);
                        double t1_ex = T2_aa_ijab(i, j, b, a);
                        double t2_ex = L2_aa_(i, j, b, a); 
                        Teff_aa(i*va_+a, j*va_+b) = 1.0 * (2.0 * t1_dir - 1.0 * t1_ex) + 1.0 * (2.0 * t2_dir - 1.0 * t2_ex);
                    }
                }
            }
        }
        Eigen::MatrixXd X_a(na_*va_, n_aux);
        X_a.noalias() = Teff_aa * B_ia_P_alpha_;
        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> XT_a(X_a.col(P).data(), va_, na_); 
            Eigen::Map<const Eigen::MatrixXd> V_a(B_vv_a.col(P).data(), va_, va_);
            Eigen::Map<const Eigen::MatrixXd> O_a(B_oo_a.col(P).data(), na_, na_);
            Z_mat_a.noalias() += V_a * XT_a - XT_a * O_a;
        }
    } 
    // =========================================================================
    // UNRESTRICTED FDF EXACT RESTORATION
    // =========================================================================
    else {
        auto* t2_bb_dense = t2_bb_.get_block(0,0,0,0);
        auto* t2_ab_dense = t2_ab_.get_block(0,0,0,0);
        Eigen::Tensor<double, 4> dummy_bb_global(nb_, nb_, vb_, vb_); dummy_bb_global.setZero();
        Eigen::Tensor<double, 4> dummy_ab_global(na_, nb_, va_, vb_); dummy_ab_global.setZero();
        if (!t2_bb_dense) t2_bb_dense = &dummy_bb_global;
        if (!t2_ab_dense) t2_ab_dense = &dummy_ab_global;

        TBLIS_VIEW_4D(t_Tbb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
        TBLIS_VIEW_4D(t_Tab, (*t2_ab_dense), na_, nb_, va_, vb_);
        
        Eigen::Tensor<double, 4> Gamma_ovov_aa(na_, va_, na_, va_); Gamma_ovov_aa.setZero();
        Eigen::Tensor<double, 4> Gamma_ovov_bb(nb_, vb_, nb_, vb_); Gamma_ovov_bb.setZero();
        Eigen::Tensor<double, 4> Gamma_ovov_ab(na_, va_, nb_, vb_); Gamma_ovov_ab.setZero();
        TBLIS_VIEW_4D(t_Govov_aa, Gamma_ovov_aa, na_, va_, na_, va_);
        TBLIS_VIEW_4D(t_Govov_bb, Gamma_ovov_bb, nb_, vb_, nb_, vb_);
        TBLIS_VIEW_4D(t_Govov_ab, Gamma_ovov_ab, na_, va_, nb_, vb_);

        tblis::mult<double>(-0.5, t_Taa, "imae", t_Taa, "jmbe", 1.0, t_Govov_aa, "iajb"); 
        tblis::mult<double>(-0.5, t_Tbb, "imae", t_Tbb, "jmbe", 1.0, t_Govov_bb, "iajb"); 
        tblis::mult<double>(-0.5, t_Tab, "miea", t_Tab, "mjeb", 1.0, t_Govov_bb, "iajb"); 
        tblis::mult<double>(-0.5, t_Taa, "imae", t_Tab, "mjeb", 1.0, t_Govov_ab, "iajb"); 
        tblis::mult<double>(-0.5, t_Tab, "imae", t_Tbb, "mjeb", 1.0, t_Govov_ab, "iajb"); 
        tblis::mult<double>(-0.5, t_Tab, "imae", t_Tab, "jmbe", 1.0, t_Govov_aa, "iajb"); 

        TBLIS_VIEW_3D(t_Bvv_b, B_vv_b.data(), vb_, vb_, n_aux);
        TBLIS_VIEW_3D(t_Boo_b, B_oo_b.data(), nb_, nb_, n_aux);
        TBLIS_VIEW_2D(t_Zb, Z_mat_b.data(), vb_, nb_);
        TBLIS_VIEW_3D(t_Bia_b, B_ia_P_beta_.data(), vb_, nb_, n_aux);

        // VVVV FDF O(N^5)
        Eigen::Tensor<double, 3> X_ij_P_a(na_, na_, n_aux);
        TBLIS_VIEW_3D(t_Xij_a, X_ij_P_a.data(), na_, na_, n_aux); 
        tblis::mult<double>(1.0, t_Taa, "ijcd", t_Bvv_a, "cdP", 0.0, t_Xij_a, "ijP");
        Eigen::Tensor<double, 3> Y_ab_P_a(va_, va_, n_aux);
        TBLIS_VIEW_3D(t_Yab_a, Y_ab_P_a.data(), va_, va_, n_aux);
        tblis::mult<double>(0.125, t_Taa, "ijab", t_Xij_a, "ijP", 0.0, t_Yab_a, "abP");
        tblis::mult<double>(4.0, t_Bia_a, "biP", t_Yab_a, "abP", 1.0, t_Za, "ai");

        Eigen::Tensor<double, 3> X_ij_P_b(nb_, nb_, n_aux);
        TBLIS_VIEW_3D(t_Xij_b, X_ij_P_b.data(), nb_, nb_, n_aux); 
        tblis::mult<double>(1.0, t_Tbb, "ijcd", t_Bvv_b, "cdP", 0.0, t_Xij_b, "ijP");
        Eigen::Tensor<double, 3> Y_ab_P_b(vb_, vb_, n_aux);
        TBLIS_VIEW_3D(t_Yab_b, Y_ab_P_b.data(), vb_, vb_, n_aux);
        tblis::mult<double>(0.125, t_Tbb, "ijab", t_Xij_b, "ijP", 0.0, t_Yab_b, "abP");
        tblis::mult<double>(4.0, t_Bia_b, "biP", t_Yab_b, "abP", 1.0, t_Zb, "ai");

        // OOOO FDF O(N^5)
        Eigen::Tensor<double, 3> X_ab_P_a(va_, va_, n_aux);
        TBLIS_VIEW_3D(t_Xab_a, X_ab_P_a.data(), va_, va_, n_aux); 
        tblis::mult<double>(1.0, t_Taa, "klab", t_Boo_a, "klP", 0.0, t_Xab_a, "abP");
        Eigen::Tensor<double, 3> Y_ij_P_a(na_, na_, n_aux);
        TBLIS_VIEW_3D(t_Yij_a, Y_ij_P_a.data(), na_, na_, n_aux);
        tblis::mult<double>(0.125, t_Taa, "ijab", t_Xab_a, "abP", 0.0, t_Yij_a, "ijP");
        tblis::mult<double>(-4.0, t_Yij_a, "ijP", t_Bia_a, "ajP", 1.0, t_Za, "ai");

        Eigen::Tensor<double, 3> X_ab_P_b(vb_, vb_, n_aux);
        TBLIS_VIEW_3D(t_Xab_b, X_ab_P_b.data(), vb_, vb_, n_aux); 
        tblis::mult<double>(1.0, t_Tbb, "klab", t_Boo_b, "klP", 0.0, t_Xab_b, "abP");
        Eigen::Tensor<double, 3> Y_ij_P_b(nb_, nb_, n_aux);
        TBLIS_VIEW_3D(t_Yij_b, Y_ij_P_b.data(), nb_, nb_, n_aux);
        tblis::mult<double>(0.125, t_Tbb, "ijab", t_Xab_b, "abP", 0.0, t_Yij_b, "ijP");
        tblis::mult<double>(-4.0, t_Yij_b, "ijP", t_Bia_b, "ajP", 1.0, t_Zb, "ai");

        Eigen::MatrixXd G_mat_aa(na_ * va_, na_ * va_);
        Eigen::MatrixXd G_mat_bb(nb_ * vb_, nb_ * vb_);
        Eigen::MatrixXd G_mat_ab(na_ * va_, nb_ * vb_);
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=0; i<na_; ++i) for(int a=0; a<va_; ++a) for(int j=0; j<na_; ++j) for(int b=0; b<va_; ++b)
            G_mat_aa(i * va_ + a, j * va_ + b) = Gamma_ovov_aa(i, a, j, b);
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=0; i<nb_; ++i) for(int a=0; a<vb_; ++a) for(int j=0; j<nb_; ++j) for(int b=0; b<vb_; ++b)
            G_mat_bb(i * vb_ + a, j * vb_ + b) = Gamma_ovov_bb(i, a, j, b);
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=0; i<na_; ++i) for(int a=0; a<va_; ++a) for(int j=0; j<nb_; ++j) for(int b=0; b<vb_; ++b)
            G_mat_ab(i * va_ + a, j * vb_ + b) = Gamma_ovov_ab(i, a, j, b);
        
        Eigen::MatrixXd Y_aa = G_mat_aa * B_ia_P_alpha_;
        TBLIS_VIEW_3D(t_Y_aa, Y_aa.data(), va_, na_, n_aux);
        tblis::mult<double>(-2.0, t_Y_aa, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");
        tblis::mult<double>( 2.0, t_Y_aa, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");

        Eigen::MatrixXd Y_bb = G_mat_bb * B_ia_P_beta_;
        TBLIS_VIEW_3D(t_Y_bb, Y_bb.data(), vb_, nb_, n_aux);
        tblis::mult<double>(-2.0, t_Y_bb, "amP", t_Boo_b, "miP", 1.0, t_Zb, "bi");
        tblis::mult<double>( 2.0, t_Y_bb, "eiP", t_Bvv_b, "aeP", 1.0, t_Zb, "bi");

        Eigen::MatrixXd Y_ab_a = G_mat_ab * B_ia_P_beta_;
        TBLIS_VIEW_3D(t_Y_ab_a, Y_ab_a.data(), va_, na_, n_aux);
        tblis::mult<double>(-2.0, t_Y_ab_a, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");
        tblis::mult<double>( 2.0, t_Y_ab_a, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");

        Eigen::MatrixXd Y_ab_b = G_mat_ab.transpose() * B_ia_P_alpha_;
        TBLIS_VIEW_3D(t_Y_ab_b, Y_ab_b.data(), vb_, nb_, n_aux);
        tblis::mult<double>(-2.0, t_Y_ab_b, "bmP", t_Boo_b, "mjP", 1.0, t_Zb, "bj");
        tblis::mult<double>( 2.0, t_Y_ab_b, "ejP", t_Bvv_b, "beP", 1.0, t_Zb, "bj");

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Teff_aa(na_*va_, na_*va_);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < na_; ++j) {
                    for (int b = 0; b < va_; ++b) {
                        Teff_aa(i*va_+a, j*va_+b) = 1.0 * T2_aa_ijab(i, j, a, b) + 1.0 * L2_aa_(i, j, a, b);
                    }
                }
            }
        }
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Teff_ab(na_*va_, nb_*vb_);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Teff_bb(nb_*vb_, nb_*vb_);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < na_; ++i) {
            for (int a = 0; a < va_; ++a) {
                for (int j = 0; j < nb_; ++j) {
                    for (int b = 0; b < vb_; ++b) {
                        Teff_ab(i*va_+a, j*vb_+b) = 1.0 * (*t2_ab_dense)(i, j, a, b) + 1.0 * L2_ab_(i, j, a, b); 
                    }
                }
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nb_; ++i) {
            for (int a = 0; a < vb_; ++a) {
                for (int j = 0; j < nb_; ++j) {
                    for (int b = 0; b < vb_; ++b) {
                        Teff_bb(i*vb_+a, j*vb_+b) = 1.0 * (*t2_bb_dense)(i, j, a, b) + 1.0 * L2_bb_(i, j, a, b); 
                    }
                }
            }
        }

        Eigen::MatrixXd X_a(na_*va_, n_aux);
        X_a.noalias() = Teff_aa * B_ia_P_alpha_;
        X_a.noalias() += Teff_ab * B_ia_P_beta_;

        Eigen::MatrixXd X_b(nb_*vb_, n_aux);
        X_b.noalias() = Teff_bb * B_ia_P_beta_;
        X_b.noalias() += Teff_ab.transpose() * B_ia_P_alpha_;

        for (int P = 0; P < n_aux; ++P) {
            Eigen::Map<const Eigen::MatrixXd> XT_a(X_a.col(P).data(), va_, na_); 
            Eigen::Map<const Eigen::MatrixXd> V_a(B_vv_a.col(P).data(), va_, va_);
            Eigen::Map<const Eigen::MatrixXd> O_a(B_oo_a.col(P).data(), na_, na_);
            Z_mat_a.noalias() += V_a * XT_a - XT_a * O_a;

            Eigen::Map<const Eigen::MatrixXd> XT_b(X_b.col(P).data(), vb_, nb_); 
            Eigen::Map<const Eigen::MatrixXd> V_b(B_vv_b.col(P).data(), vb_, vb_);
            Eigen::Map<const Eigen::MatrixXd> O_b(B_oo_b.col(P).data(), nb_, nb_);
            Z_mat_b.noalias() += V_b * XT_b - XT_b * O_b;
        }
    }
    
    F_gen_a_.block(na_, 0, va_, na_) += Z_mat_a;
    F_gen_a_.block(0, na_, na_, va_) += Z_mat_a.transpose();
    
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        F_gen_b_.block(nb_, 0, vb_, nb_) += Z_mat_b;
        F_gen_b_.block(0, nb_, nb_, vb_) += Z_mat_b.transpose();
    }
}"""

    new_content = content[:start_idx] + restored_code + "\n" + content[end_idx:]
    
    with open(filepath, 'w') as file:
        file.write(new_content)
        
    print("[+] RESTORASI MURNI FDF OMP3 BERHASIL!")

if __name__ == "__main__":
    restore_pure_omp3()