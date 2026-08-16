import re
import os
import sys

def inject_omp3_fixes(filepath="src/mp3/mp3.cc"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan. Pastikan Anda berada di root directory.")
        sys.exit(1)

    with open(filepath, 'r') as file:
        content = file.read()

    print(f"[*] Membaca file: {filepath}")
    original_length = len(content)

    # =========================================================================
    # 1. Fix OPDM Alpha (Restricted)
    # =========================================================================
    content = re.sub(
        r'tblis::mult<double>\(-0\.5, t_T2, "ikab", t_L2t, "jkab", 1\.0, t_Goo, "ij"\);\s*'
        r'tblis::mult<double>\(-0\.5, t_L2, "ikab", t_T2t, "jkab", 1\.0, t_Goo, "ij"\);',
        'tblis::mult<double>(-1.0, t_T2, "ikab", t_L2t, "jkab", 1.0, t_Goo, "ij");\n        '
        'tblis::mult<double>(-1.0, t_L2, "ikab", t_T2t, "jkab", 1.0, t_Goo, "ij");',
        content
    )
    content = re.sub(
        r'tblis::mult<double>\(0\.5, t_T2, "ijac", t_L2t, "ijbc", 1\.0, t_Gvv, "ab"\);\s*'
        r'tblis::mult<double>\(0\.5, t_L2, "ijac", t_T2t, "ijbc", 1\.0, t_Gvv, "ab"\);',
        'tblis::mult<double>(1.0, t_T2, "ijac", t_L2t, "ijbc", 1.0, t_Gvv, "ab");\n        '
        'tblis::mult<double>(1.0, t_L2, "ijac", t_T2t, "ijbc", 1.0, t_Gvv, "ab");',
        content
    )

    # =========================================================================
    # 2. Fix OPDM Alpha & Beta (Unrestricted) - Skala 2.0 untuk T3/L2
    # =========================================================================
    # Alpha T3 terms (-0.25 -> -0.5, 0.25 -> 0.5)
    content = re.sub(r'tblis::mult<double>\(-0\.25, t_T2aa, "ikab", t_T3aa', r'tblis::mult<double>(-0.5, t_T2aa, "ikab", t_T3aa', content)
    content = re.sub(r'tblis::mult<double>\(-0\.25, t_T3aa, "ikab", t_T2aa', r'tblis::mult<double>(-0.5, t_T3aa, "ikab", t_T2aa', content)
    content = re.sub(r'tblis::mult<double>\(0\.25, t_T2aa, "ijac", t_T3aa', r'tblis::mult<double>(0.5, t_T2aa, "ijac", t_T3aa', content)
    content = re.sub(r'tblis::mult<double>\(0\.25, t_T3aa, "ijac", t_T2aa', r'tblis::mult<double>(0.5, t_T3aa, "ijac", t_T2aa', content)
    
    # Cross-spin (ab) T3 terms in Alpha OPDM (-0.5 -> -1.0, 0.5 -> 1.0)
    content = re.sub(r'tblis::mult<double>\(-0\.5, t_T2ab, "ikab", t_T3ab, "jkab", 1\.0, t_Goo_a', r'tblis::mult<double>(-1.0, t_T2ab, "ikab", t_T3ab, "jkab", 1.0, t_Goo_a', content)
    content = re.sub(r'tblis::mult<double>\(-0\.5, t_T3ab, "ikab", t_T2ab, "jkab", 1\.0, t_Goo_a', r'tblis::mult<double>(-1.0, t_T3ab, "ikab", t_T2ab, "jkab", 1.0, t_Goo_a', content)
    content = re.sub(r'tblis::mult<double>\(0\.5, t_T2ab, "ijac", t_T3ab, "ijbc", 1\.0, t_Gvv_a', r'tblis::mult<double>(1.0, t_T2ab, "ijac", t_T3ab, "ijbc", 1.0, t_Gvv_a', content)
    content = re.sub(r'tblis::mult<double>\(0\.5, t_T3ab, "ijac", t_T2ab, "ijbc", 1\.0, t_Gvv_a', r'tblis::mult<double>(1.0, t_T3ab, "ijac", t_T2ab, "ijbc", 1.0, t_Gvv_a', content)

    # Beta T3 terms (-0.25 -> -0.5, 0.25 -> 0.5)
    content = re.sub(r'tblis::mult<double>\(-0\.25, t_T2bb, "ikab", t_T3bb', r'tblis::mult<double>(-0.5, t_T2bb, "ikab", t_T3bb', content)
    content = re.sub(r'tblis::mult<double>\(-0\.25, t_T3bb, "ikab", t_T2bb', r'tblis::mult<double>(-0.5, t_T3bb, "ikab", t_T2bb', content)
    content = re.sub(r'tblis::mult<double>\(0\.25, t_T2bb, "ijac", t_T3bb', r'tblis::mult<double>(0.5, t_T2bb, "ijac", t_T3bb', content)
    content = re.sub(r'tblis::mult<double>\(0\.25, t_T3bb, "ijac", t_T2bb', r'tblis::mult<double>(0.5, t_T3bb, "ijac", t_T2bb', content)

    # Cross-spin (ab) T3 terms in Beta OPDM (-0.5 -> -1.0, 0.5 -> 1.0)
    content = re.sub(r'tblis::mult<double>\(-0\.5, t_T2ab, "kiab", t_T3ab, "kjab", 1\.0, t_Goo_b', r'tblis::mult<double>(-1.0, t_T2ab, "kiab", t_T3ab, "kjab", 1.0, t_Goo_b', content)
    content = re.sub(r'tblis::mult<double>\(-0\.5, t_T3ab, "kiab", t_T2ab, "kjab", 1\.0, t_Goo_b', r'tblis::mult<double>(-1.0, t_T3ab, "kiab", t_T2ab, "kjab", 1.0, t_Goo_b', content)
    content = re.sub(r'tblis::mult<double>\(0\.5, t_T2ab, "ijca", t_T3ab, "ijcb", 1\.0, t_Gvv_b', r'tblis::mult<double>(1.0, t_T2ab, "ijca", t_T3ab, "ijcb", 1.0, t_Gvv_b', content)
    content = re.sub(r'tblis::mult<double>\(0\.5, t_T3ab, "ijca", t_T2ab, "ijcb", 1\.0, t_Gvv_b', r'tblis::mult<double>(1.0, t_T3ab, "ijca", t_T2ab, "ijcb", 1.0, t_Gvv_b', content)

    # =========================================================================
    # 3. Fix Teff (Generalized Fock) Scaling
    # =========================================================================
    # Restricted Teff
    content = re.sub(
        r'Teff_aa\(i\*va_\+a, j\*va_\+b\) = 1\.0 \* \(2\.0 \* t1_dir - 1\.0 \* t1_ex\) \+ 1\.0 \* \(2\.0 \* t2_dir - 1\.0 \* t2_ex\);',
        r'Teff_aa(i*va_+a, j*va_+b) = 1.0 * (2.0 * t1_dir - 1.0 * t1_ex) + 2.0 * (2.0 * t2_dir - 1.0 * t2_ex);',
        content
    )
    # Unrestricted Teff
    content = re.sub(r'1\.0 \* L2_aa_\(i, j, a, b\);', r'2.0 * L2_aa_(i, j, a, b);', content)
    content = re.sub(r'1\.0 \* L2_ab_\(i, j, a, b\);', r'2.0 * L2_ab_(i, j, a, b);', content)
    content = re.sub(r'1\.0 \* L2_bb_\(i, j, a, b\);', r'2.0 * L2_bb_(i, j, a, b);', content)

    # =========================================================================
    # 4. Fix FDF Segfault (Bvv -> Bia & Remove cross-spin AB block)
    # =========================================================================
    vvvv_unrestricted_old = r"""        Eigen::Tensor<double, 3> X_ij_P_a\(na_, na_, n_aux\);
        TBLIS_VIEW_3D\(t_Xij_a, X_ij_P_a\.data\(\), na_, na_, n_aux\); 
        tblis::mult<double>\(1\.0, t_Taa, "ijcd", t_Bvv_a, "cdP", 0\.0, t_Xij_a, "ijP"\);
        Eigen::Tensor<double, 3> Y_ab_P_a\(va_, va_, n_aux\);
        TBLIS_VIEW_3D\(t_Yab_a, Y_ab_P_a\.data\(\), va_, va_, n_aux\);
        tblis::mult<double>\(0\.125, t_Taa, "ijab", t_Xij_a, "ijP", 0\.0, t_Yab_a, "abP"\);
        tblis::mult<double>\(4\.0, t_Bvv_a, "ebP", t_Yab_a, "abP", 1\.0, t_Za, "ae"\);

        Eigen::Tensor<double, 3> X_ij_P_b\(nb_, nb_, n_aux\);
        TBLIS_VIEW_3D\(t_Xij_b, X_ij_P_b\.data\(\), nb_, nb_, n_aux\); 
        tblis::mult<double>\(1\.0, t_Tbb, "ijcd", t_Bvv_b, "cdP", 0\.0, t_Xij_b, "ijP"\);
        Eigen::Tensor<double, 3> Y_ab_P_b\(vb_, vb_, n_aux\);
        TBLIS_VIEW_3D\(t_Yab_b, Y_ab_P_b\.data\(\), vb_, vb_, n_aux\);
        tblis::mult<double>\(0\.125, t_Tbb, "ijab", t_Xij_b, "ijP", 0\.0, t_Yab_b, "abP"\);
        tblis::mult<double>\(4\.0, t_Bvv_b, "ebP", t_Yab_b, "abP", 1\.0, t_Zb, "ae"\);

        Eigen::Tensor<double, 3> X_ij_P_ab\(na_, nb_, n_aux\);
        TBLIS_VIEW_3D\(t_Xij_ab, X_ij_P_ab\.data\(\), na_, nb_, n_aux\); 
        tblis::mult<double>\(1\.0, t_Tab, "ijcd", t_Bvv_b, "cdP", 0\.0, t_Xij_ab, "ijP"\);
        Eigen::Tensor<double, 3> Y_ab_P_ab\(va_, vb_, n_aux\);
        TBLIS_VIEW_3D\(t_Yab_ab, Y_ab_P_ab\.data\(\), va_, vb_, n_aux\);
        tblis::mult<double>\(1\.0, t_Tab, "ijab", t_Xij_ab, "ijP", 0\.0, t_Yab_ab, "abP"\);
        tblis::mult<double>\(4\.0, t_Bvv_a, "ebP", t_Yab_ab, "abP", 1\.0, t_Za, "ae"\);"""

    vvvv_unrestricted_new = """        Eigen::Tensor<double, 3> X_ij_P_a(na_, na_, n_aux);
        TBLIS_VIEW_3D(t_Xij_a, X_ij_P_a.data(), na_, na_, n_aux); 
        tblis::mult<double>(1.0, t_Taa, "ijcd", t_Bvv_a, "cdP", 0.0, t_Xij_a, "ijP");
        Eigen::Tensor<double, 3> Y_ab_P_a(va_, va_, n_aux);
        TBLIS_VIEW_3D(t_Yab_a, Y_ab_P_a.data(), va_, va_, n_aux);
        tblis::mult<double>(0.125, t_Taa, "ijab", t_Xij_a, "ijP", 0.0, t_Yab_a, "abP");
        tblis::mult<double>(4.0, t_Bia_a, "biP", t_Yab_a, "abP", 1.0, t_Za, "ai"); // Diperbaiki ke Bia & ai

        Eigen::Tensor<double, 3> X_ij_P_b(nb_, nb_, n_aux);
        TBLIS_VIEW_3D(t_Xij_b, X_ij_P_b.data(), nb_, nb_, n_aux); 
        tblis::mult<double>(1.0, t_Tbb, "ijcd", t_Bvv_b, "cdP", 0.0, t_Xij_b, "ijP");
        Eigen::Tensor<double, 3> Y_ab_P_b(vb_, vb_, n_aux);
        TBLIS_VIEW_3D(t_Yab_b, Y_ab_P_b.data(), vb_, vb_, n_aux);
        tblis::mult<double>(0.125, t_Tbb, "ijab", t_Xij_b, "ijP", 0.0, t_Yab_b, "abP");
        tblis::mult<double>(4.0, t_Bia_b, "biP", t_Yab_b, "abP", 1.0, t_Zb, "ai"); // Diperbaiki ke Bia & ai"""
    
    content = re.sub(vvvv_unrestricted_old, vvvv_unrestricted_new, content)

    # Hapus blok AB di OOOO
    oooo_ab_old = r"""\n\s*Eigen::Tensor<double, 3> X_ab_P_ab\(va_, vb_, n_aux\);
\s*TBLIS_VIEW_3D\(t_Xab_ab, X_ab_P_ab\.data\(\), va_, vb_, n_aux\); 
\s*tblis::mult<double>\(1\.0, t_Tab, "klab", t_Boo_b, "klP", 0\.0, t_Xab_ab, "abP"\);
\s*Eigen::Tensor<double, 3> Y_ij_P_ab\(na_, nb_, n_aux\);
\s*TBLIS_VIEW_3D\(t_Yij_ab, Y_ij_P_ab\.data\(\), na_, nb_, n_aux\);
\s*tblis::mult<double>\(1\.0, t_Tab, "ijab", t_Xab_ab, "abP", 0\.0, t_Yij_ab, "ijP"\);
\s*tblis::mult<double>\(-4\.0, t_Yij_ab, "ijP", t_Bia_a, "ajP", 1\.0, t_Za, "ai"\);"""
    
    content = re.sub(oooo_ab_old, "", content)

    if len(content) == original_length:
        print("[-] Tidak ada perubahan yang dilakukan. Mungkin script sudah pernah dijalankan sebelumnya.")
    else:
        with open(filepath, 'w') as file:
            file.write(content)
        print("[+] Sukses! Injection selesai diterapkan ke dalam src/mp3/mp3.cc.")

if __name__ == "__main__":
    inject_omp3_fixes()