import os, re, sys

FILE_PATH = "src/mp3/mp3.cc"

def patch_final():
    if not os.path.exists(FILE_PATH):
        print(f"[FATAL] Fail {FILE_PATH} tidak ditemukan.")
        sys.exit(1)
        
    with open(FILE_PATH, 'r') as f:
        c = f.read()

    # 1. MEMBEDAH SCRAMBLED INDICES (Mengembalikan TBLIS ke Trajektori Fisika)
    # Restricted
    c = re.sub(r't_Tt,\s*"ijac",\s*t_T,\s*"ijbd"', r't_Tt, "ijab", t_T, "ijcd"', c)
    c = re.sub(r't_Tt,\s*"ikab",\s*t_T,\s*"jlab"', r't_Tt, "ijab", t_T, "klab"', c)
    # Unrestricted AA & BB
    c = re.sub(r't_Tleft,\s*"ijac",\s*t_Tright,\s*"ijbd"', r't_Tleft, "ijab", t_Tright, "ijcd"', c)
    c = re.sub(r't_Tleft,\s*"ikab",\s*t_Tright,\s*"jlab"', r't_Tleft, "ijab", t_Tright, "klab"', c)
    
    # 2. INJEKSI FAKTOR TOPOLOGI (Permutasi Simetri Matriks-Z)
    c = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_Gvvvv_s', r'tblis::mult<double>( 4.0, t_Gvvvv_s', c)
    c = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_Gvvvv_b_s', r'tblis::mult<double>( 4.0, t_Gvvvv_b_s', c)
    c = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_Goooo_s', r'tblis::mult<double>(-4.0, t_Goooo_s', c)
    c = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_Goooo_b_s', r'tblis::mult<double>(-4.0, t_Goooo_b_s', c)
    
    # 3. KOREKSI REDUKSI OVOV
    c = re.sub(r'tblis::mult<double>\(\s*-0\.25\*scale,\s*t_Tleft,\s*"imae",\s*t_Tright,\s*"jmbe",\s*1\.0,\s*t_Govov_aa,\s*"iajb"\);', 
               r'tblis::mult<double>(-0.5*scale,  t_Tleft, "imae", t_Tright, "jmbe", 1.0, t_Govov_aa, "iajb");', c)
    c = re.sub(r'tblis::mult<double>\(\s*-0\.25\*scale,\s*t_Tleft,\s*"imae",\s*t_Tright,\s*"jmbe",\s*1\.0,\s*t_Govov_bb,\s*"iajb"\);', 
               r'tblis::mult<double>(-0.5*scale,  t_Tleft, "imae", t_Tright, "jmbe", 1.0, t_Govov_bb, "iajb");', c)

    with open(FILE_PATH, 'w') as f:
        f.write(c)
        
    print("[METRIK] Injeksi Indeks TBLIS dan Topologi Lagrangian Sukses!")

if __name__ == "__main__":
    patch_final()