import os
import sys

FILE_PATH = "src/mp3/mp3.cc"

def fix_tblis_indices():
    if not os.path.exists(FILE_PATH):
        print(f"[FATAL] Fail {FILE_PATH} tidak ditemukan.")
        sys.exit(1)

    with open(FILE_PATH, 'r') as f:
        content = f.read()

    # Koreksi Blok Restricted AA
    content = content.replace(
        'tblis::mult<double>( 0.25*scale, t_Tt, "ijac", t_T, "ijbd", 1.0, t_Gvvvv_aa, "abcd");',
        'tblis::mult<double>( 0.25*scale, t_Tt, "ijab", t_T, "ijcd", 1.0, t_Gvvvv_aa, "abcd");'
    )
    content = content.replace(
        'tblis::mult<double>( 0.25*scale, t_Tt, "ikab", t_T, "jlab", 1.0, t_Goooo_aa, "ijkl");',
        'tblis::mult<double>( 0.25*scale, t_Tt, "ijab", t_T, "klab", 1.0, t_Goooo_aa, "ijkl");'
    )

    # Koreksi Blok Unrestricted AA
    content = content.replace(
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ijac", t_Tright, "ijbd", 1.0, t_Gvvvv_aa, "abcd");',
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ijab", t_Tright, "ijcd", 1.0, t_Gvvvv_aa, "abcd");'
    )
    content = content.replace(
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ikab", t_Tright, "jlab", 1.0, t_Goooo_aa, "ijkl");',
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ijab", t_Tright, "klab", 1.0, t_Goooo_aa, "ijkl");'
    )

    # Koreksi Blok Unrestricted BB
    content = content.replace(
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ijac", t_Tright, "ijbd", 1.0, t_Gvvvv_bb, "abcd");',
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ijab", t_Tright, "ijcd", 1.0, t_Gvvvv_bb, "abcd");'
    )
    content = content.replace(
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ikab", t_Tright, "jlab", 1.0, t_Goooo_bb, "ijkl");',
        'tblis::mult<double>( 0.125*scale, t_Tleft, "ijab", t_Tright, "klab", 1.0, t_Goooo_bb, "ijkl");'
    )

    with open(FILE_PATH, 'w') as f:
        f.write(content)
    
    print("[METRIK] Koreksi indeks TBLIS 2-RDM sukses diaplikasikan.")

if __name__ == "__main__":
    fix_tblis_indices()