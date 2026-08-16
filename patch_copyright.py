import os
import sys

FILE_PATH = "src/mp3/mp3.cc"

def apply_topological_factors():
    if not os.path.exists(FILE_PATH):
        print(f"[FATAL] Fail {FILE_PATH} tidak ditemukan.")
        sys.exit(1)

    with open(FILE_PATH, 'r') as f:
        content = f.read()

    # 1. Faktor Topologis 4.0 untuk blok VVVV (Alpha & Beta)
    content = content.replace(
        'tblis::mult<double>( 1.0, t_Gvvvv_s, "abcd", t_Bvv_a, "cdP", 0.0, t_Xvv_a, "abP");',
        'tblis::mult<double>( 4.0, t_Gvvvv_s, "abcd", t_Bvv_a, "cdP", 0.0, t_Xvv_a, "abP");'
    )
    content = content.replace(
        'tblis::mult<double>( 1.0, t_Gvvvv_b_s, "abcd", t_Bvv_b, "cdP", 0.0, t_Xvv_b, "abP");',
        'tblis::mult<double>( 4.0, t_Gvvvv_b_s, "abcd", t_Bvv_b, "cdP", 0.0, t_Xvv_b, "abP");'
    )

    # 2. Faktor Topologis 4.0 untuk blok OOOO (Alpha & Beta)
    content = content.replace(
        'tblis::mult<double>(-1.0, t_Goooo_s, "ijkl", t_Boo_a, "klP", 0.0, t_Xoo_a, "ijP");',
        'tblis::mult<double>(-4.0, t_Goooo_s, "ijkl", t_Boo_a, "klP", 0.0, t_Xoo_a, "ijP");'
    )
    content = content.replace(
        'tblis::mult<double>(-1.0, t_Goooo_b_s, "ijkl", t_Boo_b, "klP", 0.0, t_Xoo_b, "ijP");',
        'tblis::mult<double>(-4.0, t_Goooo_b_s, "ijkl", t_Boo_b, "klP", 0.0, t_Xoo_b, "ijP");'
    )

    # 3. Restorasi Faktor Topologis 2.0 untuk blok OVOV (Alpha-Alpha & Beta-Beta)
    # Merubah pengali dari -0.25 (yang tereduksi) menjadi -0.5 untuk menyeimbangkan Z_mat
    content = content.replace(
        'tblis::mult<double>(-0.25*scale,  t_Tleft, "imae", t_Tright, "jmbe", 1.0, t_Govov_aa, "iajb");',
        'tblis::mult<double>(-0.5*scale,  t_Tleft, "imae", t_Tright, "jmbe", 1.0, t_Govov_aa, "iajb");'
    )
    content = content.replace(
        'tblis::mult<double>(-0.25*scale,  t_Tleft, "imae", t_Tright, "jmbe", 1.0, t_Govov_bb, "iajb");',
        'tblis::mult<double>(-0.5*scale,  t_Tleft, "imae", t_Tright, "jmbe", 1.0, t_Govov_bb, "iajb");'
    )

    with open(FILE_PATH, 'w') as f:
        f.write(content)
    
    print("[METRIK] Injeksi Faktor Topologis 2-RDM MP3 sukses diaplikasikan.")

if __name__ == "__main__":
    apply_topological_factors()