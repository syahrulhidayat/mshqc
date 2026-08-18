import os

def patch_omp3_cross_terms(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} tidak ditemukan!")
        return

    with open(filepath, 'r') as f:
        code = f.read()

    # ==========================================
    # BLOK 1: RESTRICTED - T2_tilde & Govov
    # ==========================================
    target_1 = """        Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);
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

        Eigen::Tensor<double, 4> Gamma_ovov_aa(na_, va_, na_, va_); Gamma_ovov_aa.setZero();
        TBLIS_VIEW_4D(t_Govov_aa, Gamma_ovov_aa, na_, va_, na_, va_);
        
        // Restorasi tanda: Positif untuk kontraksi langsung, negatif untuk pertukaran (exchange)
        tblis::mult<double>(1.0, t_T2t, "imae", t_Taa, "jmeb", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(1.0, t_T2t, "mjea", t_Taa, "mibe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-0.5, t_T2t, "mjea", t_Taa, "miba", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-0.5, t_T2t, "imae", t_Taa, "jmba", 1.0, t_Govov_aa, "iajb");"""

    replace_1 = """        // 1. Buat T2_tilde dan T3_tilde untuk cross-terms
        Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);
        Eigen::Tensor<double, 4> T3_tilde(na_, na_, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for(int i=0; i<na_; ++i) {
            for(int j=0; j<na_; ++j) {
                for(int a=0; a<va_; ++a) {
                    for(int b=0; b<va_; ++b) {
                        T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);
                        T3_tilde(i,j,a,b) = 2.0 * L2_aa_(i,j,a,b) - L2_aa_(i,j,b,a);
                    }
                }
            }
        }
        TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);
        TBLIS_VIEW_4D(t_T3t, T3_tilde, na_, na_, va_, va_);

        Eigen::Tensor<double, 4> Gamma_ovov_aa(na_, va_, na_, va_); Gamma_ovov_aa.setZero();
        TBLIS_VIEW_4D(t_Govov_aa, Gamma_ovov_aa, na_, va_, na_, va_);
        
        // 2. Blok Ring (Govov) - Restorasi tanda ke NEGATIF dan injeksi Cross-Terms MP3
        // [MP2] T1 * T1
        tblis::mult<double>(-1.0, t_T2t, "imae", t_Taa, "jmeb", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-1.0, t_T2t, "mjea", t_Taa, "mibe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(0.5, t_T2t, "mjea", t_Taa, "miba", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(0.5, t_T2t, "imae", t_Taa, "jmba", 1.0, t_Govov_aa, "iajb");
        // [MP3] T1 * T2
        tblis::mult<double>(-1.0, t_T2t, "imae", t_T3aa, "jmeb", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-1.0, t_T2t, "mjea", t_T3aa, "mibe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(0.5, t_T2t, "mjea", t_T3aa, "miba", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(0.5, t_T2t, "imae", t_T3aa, "jmba", 1.0, t_Govov_aa, "iajb");
        // [MP3] T2 * T1
        tblis::mult<double>(-1.0, t_T3t, "imae", t_Taa, "jmeb", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-1.0, t_T3t, "mjea", t_Taa, "mibe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(0.5, t_T3t, "mjea", t_Taa, "miba", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(0.5, t_T3t, "imae", t_Taa, "jmba", 1.0, t_Govov_aa, "iajb");"""

    # ==========================================
    # BLOK 2: RESTRICTED - Gvvvv
    # ==========================================
    target_2 = """        // KOREKSI FINAL: Skalar 2-RDM Gvvvv direduksi secara proporsional (0.5)
        tblis::mult<double>(0.5, t_T2t, "ijcb", t_Taa, "ijad", 0.0, t_Gvvvv_a, "acbd");
        tblis::mult<double>(0.5, t_T2t, "ijbc", t_Taa, "ijda", 1.0, t_Gvvvv_a, "acbd");"""

    replace_2 = """        // 3. Blok Ladder Virtual (Gvvvv) - Injeksi Cross-Terms
        tblis::mult<double>(0.5, t_T2t, "ijcb", t_Taa, "ijad", 0.0, t_Gvvvv_a, "acbd");
        tblis::mult<double>(0.5, t_T2t, "ijbc", t_Taa, "ijda", 1.0, t_Gvvvv_a, "acbd");
        tblis::mult<double>(0.5, t_T2t, "ijcb", t_T3aa, "ijad", 1.0, t_Gvvvv_a, "acbd");
        tblis::mult<double>(0.5, t_T2t, "ijbc", t_T3aa, "ijda", 1.0, t_Gvvvv_a, "acbd");
        tblis::mult<double>(0.5, t_T3t, "ijcb", t_Taa, "ijad", 1.0, t_Gvvvv_a, "acbd");
        tblis::mult<double>(0.5, t_T3t, "ijbc", t_Taa, "ijda", 1.0, t_Gvvvv_a, "acbd");"""

    # ==========================================
    # BLOK 3: RESTRICTED - Goooo
    # ==========================================
    target_3 = """        // KOREKSI FINAL: Skalar 2-RDM Goooo direduksi secara proporsional (0.5)
        tblis::mult<double>(0.5, t_T2t, "inab", t_Taa, "klab", 0.0, t_Goooo_a, "ikln");
        tblis::mult<double>(0.5, t_T2t, "lnab", t_Taa, "kiab", 1.0, t_Goooo_a, "ikln");"""

    replace_3 = """        // 4. Blok Ladder Occupied (Goooo) - Injeksi Cross-Terms
        tblis::mult<double>(0.5, t_T2t, "inab", t_Taa, "klab", 0.0, t_Goooo_a, "ikln");
        tblis::mult<double>(0.5, t_T2t, "lnab", t_Taa, "kiab", 1.0, t_Goooo_a, "ikln");
        tblis::mult<double>(0.5, t_T2t, "inab", t_T3aa, "klab", 1.0, t_Goooo_a, "ikln");
        tblis::mult<double>(0.5, t_T2t, "lnab", t_T3aa, "kiab", 1.0, t_Goooo_a, "ikln");
        tblis::mult<double>(0.5, t_T3t, "inab", t_Taa, "klab", 1.0, t_Goooo_a, "ikln");
        tblis::mult<double>(0.5, t_T3t, "lnab", t_Taa, "kiab", 1.0, t_Goooo_a, "ikln");"""

    # ==========================================
    # BLOK 4: UNRESTRICTED - Govov
    # ==========================================
    target_4 = """        /// Restorasi tanda: Positif seragam untuk Govov Unrestricted
        tblis::mult<double>(1.0, t_Taa, "imae", t_Taa, "jmbe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(1.0, t_Tbb, "imae", t_Tbb, "jmbe", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(1.0, t_Tab, "miea", t_Tab, "mjeb", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(1.0, t_Taa, "imae", t_Tab, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(1.0, t_Tab, "imae", t_Tbb, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(1.0, t_Tab, "imae", t_Tab, "jmbe", 1.0, t_Govov_aa, "iajb");"""

    replace_4 = """        // 1. Blok Ring (Govov) Unrestricted - Restorasi tanda ke NEGATIF dan injeksi Cross-Terms
        // [MP2]
        tblis::mult<double>(-1.0, t_Taa, "imae", t_Taa, "jmbe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-1.0, t_Tbb, "imae", t_Tbb, "jmbe", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(-1.0, t_Tab, "miea", t_Tab, "mjeb", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(-1.0, t_Taa, "imae", t_Tab, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(-1.0, t_Tab, "imae", t_Tbb, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(-1.0, t_Tab, "imae", t_Tab, "jmbe", 1.0, t_Govov_aa, "iajb");
        // [MP3] T1 * T2
        tblis::mult<double>(-1.0, t_Taa, "imae", t_T3aa, "jmbe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-1.0, t_Tbb, "imae", t_T3bb, "jmbe", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(-1.0, t_Tab, "miea", t_T3ab, "mjeb", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(-1.0, t_Taa, "imae", t_T3ab, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(-1.0, t_Tab, "imae", t_T3bb, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(-1.0, t_Tab, "imae", t_T3ab, "jmbe", 1.0, t_Govov_aa, "iajb");
        // [MP3] T2 * T1
        tblis::mult<double>(-1.0, t_T3aa, "imae", t_Taa, "jmbe", 1.0, t_Govov_aa, "iajb");
        tblis::mult<double>(-1.0, t_T3bb, "imae", t_Tbb, "jmbe", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(-1.0, t_T3ab, "miea", t_Tab, "mjeb", 1.0, t_Govov_bb, "iajb");
        tblis::mult<double>(-1.0, t_T3aa, "imae", t_Tab, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(-1.0, t_T3ab, "imae", t_Tbb, "mjeb", 1.0, t_Govov_ab, "iajb");
        tblis::mult<double>(-1.0, t_T3ab, "imae", t_Tab, "jmbe", 1.0, t_Govov_aa, "iajb");"""

    # ==========================================
    # BLOK 5: UNRESTRICTED - Gvvvv
    # ==========================================
    target_5 = """        Eigen::Tensor<double, 4> Gvvvv_aa(va_, va_, va_, va_); Gvvvv_aa.setZero();
        TBLIS_VIEW_4D(t_Gvvvv_aa, Gvvvv_aa, va_, va_, va_, va_);
        // KOREKSI FINAL: Skalar 2-RDM Gvvvv direduksi secara proporsional (0.25 dan 0.5)
        tblis::mult<double>(0.25, t_Taa, "ijcb", t_Taa, "ijad", 0.0, t_Gvvvv_aa, "acbd");
        
        Eigen::Tensor<double, 4> Gvvvv_bb(vb_, vb_, vb_, vb_); Gvvvv_bb.setZero();
        TBLIS_VIEW_4D(t_Gvvvv_bb, Gvvvv_bb, vb_, vb_, vb_, vb_);
        tblis::mult<double>(0.25, t_Tbb, "ijcb", t_Tbb, "ijad", 0.0, t_Gvvvv_bb, "acbd");
        
        Eigen::Tensor<double, 4> Gvvvv_ab(va_, va_, vb_, vb_); Gvvvv_ab.setZero();
        TBLIS_VIEW_4D(t_Gvvvv_ab, Gvvvv_ab, va_, va_, vb_, vb_);
        tblis::mult<double>(0.5, t_Tab, "ijcb", t_Tab, "ijad", 0.0, t_Gvvvv_ab, "acbd");"""

    replace_5 = """        Eigen::Tensor<double, 4> Gvvvv_aa(va_, va_, va_, va_); Gvvvv_aa.setZero();
        TBLIS_VIEW_4D(t_Gvvvv_aa, Gvvvv_aa, va_, va_, va_, va_);
        Eigen::Tensor<double, 4> Gvvvv_bb(vb_, vb_, vb_, vb_); Gvvvv_bb.setZero();
        TBLIS_VIEW_4D(t_Gvvvv_bb, Gvvvv_bb, vb_, vb_, vb_, vb_);
        Eigen::Tensor<double, 4> Gvvvv_ab(va_, va_, vb_, vb_); Gvvvv_ab.setZero();
        TBLIS_VIEW_4D(t_Gvvvv_ab, Gvvvv_ab, va_, va_, vb_, vb_);

        // 2. Blok Ladder Virtual (Gvvvv) Unrestricted - Injeksi Cross-Terms
        // [MP2]
        tblis::mult<double>(0.25, t_Taa, "ijcb", t_Taa, "ijad", 0.0, t_Gvvvv_aa, "acbd");
        tblis::mult<double>(0.25, t_Tbb, "ijcb", t_Tbb, "ijad", 0.0, t_Gvvvv_bb, "acbd");
        tblis::mult<double>(0.5, t_Tab, "ijcb", t_Tab, "ijad", 0.0, t_Gvvvv_ab, "acbd");
        // [MP3] T1 * T2 & T2 * T1
        tblis::mult<double>(0.25, t_Taa, "ijcb", t_T3aa, "ijad", 1.0, t_Gvvvv_aa, "acbd");
        tblis::mult<double>(0.25, t_T3aa, "ijcb", t_Taa, "ijad", 1.0, t_Gvvvv_aa, "acbd");
        tblis::mult<double>(0.25, t_Tbb, "ijcb", t_T3bb, "ijad", 1.0, t_Gvvvv_bb, "acbd");
        tblis::mult<double>(0.25, t_T3bb, "ijcb", t_Tbb, "ijad", 1.0, t_Gvvvv_bb, "acbd");
        tblis::mult<double>(0.5, t_Tab, "ijcb", t_T3ab, "ijad", 1.0, t_Gvvvv_ab, "acbd");
        tblis::mult<double>(0.5, t_T3ab, "ijcb", t_Tab, "ijad", 1.0, t_Gvvvv_ab, "acbd");"""

    # ==========================================
    # BLOK 6: UNRESTRICTED - Goooo
    # ==========================================
    target_6 = """        Eigen::Tensor<double, 4> Goooo_aa(na_, na_, na_, na_); Goooo_aa.setZero();
        TBLIS_VIEW_4D(t_Goooo_aa, Goooo_aa, na_, na_, na_, na_);
        // KOREKSI FINAL: Skalar 2-RDM Goooo direduksi secara proporsional (0.25 dan 0.5)
        tblis::mult<double>(0.25, t_Taa, "inab", t_Taa, "klab", 0.0, t_Goooo_aa, "ikln");

        Eigen::Tensor<double, 4> Goooo_bb(nb_, nb_, nb_, nb_); Goooo_bb.setZero();
        TBLIS_VIEW_4D(t_Goooo_bb, Goooo_bb, nb_, nb_, nb_, nb_);
        tblis::mult<double>(0.25, t_Tbb, "inab", t_Tbb, "klab", 0.0, t_Goooo_bb, "ikln");

        Eigen::Tensor<double, 4> Goooo_ab(na_, na_, nb_, nb_); Goooo_ab.setZero();
        TBLIS_VIEW_4D(t_Goooo_ab, Goooo_ab, na_, na_, nb_, nb_);
        tblis::mult<double>(0.5, t_Tab, "inab", t_Tab, "klab", 0.0, t_Goooo_ab, "ikln");"""

    replace_6 = """        Eigen::Tensor<double, 4> Goooo_aa(na_, na_, na_, na_); Goooo_aa.setZero();
        TBLIS_VIEW_4D(t_Goooo_aa, Goooo_aa, na_, na_, na_, na_);
        Eigen::Tensor<double, 4> Goooo_bb(nb_, nb_, nb_, nb_); Goooo_bb.setZero();
        TBLIS_VIEW_4D(t_Goooo_bb, Goooo_bb, nb_, nb_, nb_, nb_);
        Eigen::Tensor<double, 4> Goooo_ab(na_, na_, nb_, nb_); Goooo_ab.setZero();
        TBLIS_VIEW_4D(t_Goooo_ab, Goooo_ab, na_, na_, nb_, nb_);

        // 3. Blok Ladder Occupied (Goooo) Unrestricted - Injeksi Cross-Terms
        // [MP2]
        tblis::mult<double>(0.25, t_Taa, "inab", t_Taa, "klab", 0.0, t_Goooo_aa, "ikln");
        tblis::mult<double>(0.25, t_Tbb, "inab", t_Tbb, "klab", 0.0, t_Goooo_bb, "ikln");
        tblis::mult<double>(0.5, t_Tab, "inab", t_Tab, "klab", 0.0, t_Goooo_ab, "ikln");
        // [MP3] T1 * T2 & T2 * T1
        tblis::mult<double>(0.25, t_Taa, "inab", t_T3aa, "klab", 1.0, t_Goooo_aa, "ikln");
        tblis::mult<double>(0.25, t_T3aa, "inab", t_Taa, "klab", 1.0, t_Goooo_aa, "ikln");
        tblis::mult<double>(0.25, t_Tbb, "inab", t_T3bb, "klab", 1.0, t_Goooo_bb, "ikln");
        tblis::mult<double>(0.25, t_T3bb, "inab", t_Tbb, "klab", 1.0, t_Goooo_bb, "ikln");
        tblis::mult<double>(0.5, t_Tab, "inab", t_T3ab, "klab", 1.0, t_Goooo_ab, "ikln");
        tblis::mult<double>(0.5, t_T3ab, "inab", t_Tab, "klab", 1.0, t_Goooo_ab, "ikln");"""

    targets = [target_1, target_2, target_3, target_4, target_5, target_6]
    replaces = [replace_1, replace_2, replace_3, replace_4, replace_5, replace_6]
    names = ["Restricted Govov", "Restricted Gvvvv", "Restricted Goooo", 
             "Unrestricted Govov", "Unrestricted Gvvvv", "Unrestricted Goooo"]

    for t, r, name in zip(targets, replaces, names):
        if t in code:
            code = code.replace(t, r)
            print(f"[+] Berhasil memodifikasi blok: {name}")
        else:
            print(f"[-] GAGAL menemukan blok: {name} (Bisa jadi karena perbedaan spasi/indentasi).")

    with open(filepath, 'w') as f:
        f.write(code)
    
    print("\nProses injeksi selesai. Silakan lakukan kompilasi (make) dan jalankan tes kembali.")

# Arahkan ke file source C++ Anda (sesuaikan path bila diperlukan)
patch_omp3_cross_terms("src/mp3/mp3.cc")