import os
import sys
import re

def fix_omp3_lsep(filepath="src/mp3/mp3.cc"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan!")
        sys.exit(1)

    with open(filepath, 'r') as file:
        content = file.read()

    # Perbaikan untuk Alpha
    old_a = r"Eigen::MatrixXd L_sep_a = G_vv_alpha_ \* F_HF_vo_a - F_HF_vo_a \* G_oo_alpha_;"
    new_a = r"Eigen::MatrixXd L_sep_a = F_HF_vo_a * G_oo_alpha_ - G_vv_alpha_ * F_HF_vo_a;"
    content = re.sub(old_a, new_a, content)

    # Perbaikan untuk Beta
    old_b = r"Eigen::MatrixXd L_sep_b = G_vv_beta_ \* F_HF_vo_b - F_HF_vo_b \* G_oo_beta_;"
    new_b = r"Eigen::MatrixXd L_sep_b = F_HF_vo_b * G_oo_beta_ - G_vv_beta_ * F_HF_vo_b;"
    content = re.sub(old_b, new_b, content)

    with open(filepath, 'w') as file:
        file.write(content)
        
    print("[+] Separable Lagrangian (L_sep) di OMP3 berhasil diperbaiki!")

if __name__ == "__main__":
    fix_omp3_lsep()