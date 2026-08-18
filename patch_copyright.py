import re
import os
import sys

def fix_omp3_tblis_views(filepath="src/mp3/mp3.cc"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan.")
        sys.exit(1)

    with open(filepath, 'r') as file:
        content = file.read()

    # 1. Deklarasi TBLIS_VIEW_4D untuk t_T3aa
    content = re.sub(
        r'(TBLIS_VIEW_4D\(t_Taa,\s*T2_aa_ijab,\s*na_,\s*na_,\s*va_,\s*va_\);)',
        r'\1\n    TBLIS_VIEW_4D(t_T3aa, L2_aa_, na_, na_, va_, va_);',
        content
    )

    # 2. Deklarasi TBLIS_VIEW_4D untuk t_T3bb dan t_T3ab
    content = re.sub(
        r'(TBLIS_VIEW_4D\(t_Tab,\s*\(\*t2_ab_dense\),\s*na_,\s*nb_,\s*va_,\s*vb_\);)',
        r'\1\n        TBLIS_VIEW_4D(t_T3bb, L2_bb_, nb_, nb_, vb_, vb_);\n        TBLIS_VIEW_4D(t_T3ab, L2_ab_, na_, nb_, va_, vb_);',
        content
    )

    # 3. Deklarasi alokasi Eigen dan TBLIS_VIEW_4D untuk t_L2t (Sistem Restricted)
    restricted_pattern = r"(Eigen::Tensor<double, 4> T2_tilde\(na_, na_, va_, va_\);[\s\S]*?T2_tilde\(i,j,a,b\) = 2\.0 \* T2_aa_ijab\(i,j,a,b\) - T2_aa_ijab\(i,j,b,a\);[\s\S]*?TBLIS_VIEW_4D\(t_T2t, T2_tilde, na_, na_, va_, va_\);)"
    
    def repl_restricted(match):
        text = match.group(1)
        text = text.replace(
            "Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);",
            "Eigen::Tensor<double, 4> T2_tilde(na_, na_, va_, va_);\n        Eigen::Tensor<double, 4> L2_tilde(na_, na_, va_, va_);"
        )
        text = text.replace(
            "T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);",
            "T2_tilde(i,j,a,b) = 2.0 * T2_aa_ijab(i,j,a,b) - T2_aa_ijab(i,j,b,a);\n                        L2_tilde(i,j,a,b) = 2.0 * L2_aa_(i,j,a,b) - L2_aa_(i,j,b,a);"
        )
        text = text.replace(
            "TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);",
            "TBLIS_VIEW_4D(t_T2t, T2_tilde, na_, na_, va_, va_);\n        TBLIS_VIEW_4D(t_L2t, L2_tilde, na_, na_, va_, va_);"
        )
        return text

    new_content = re.sub(restricted_pattern, repl_restricted, content)

    if new_content == content:
        print("[-] Tidak ada perbaikan yang diterapkan. Pastikan struktur blok tidak berubah.")
    else:
        with open(filepath, 'w') as file:
            file.write(new_content)
        print("[+] Definisi TBLIS View untuk tensor T3 (L2) berhasil ditambahkan.")

if __name__ == "__main__":
    fix_omp3_tblis_views()