import os
import re
import sys

FILE_PATH = "src/mp3/mp3.cc"

def patch_mp3_scaling():
    print(f"[METRIK] Memindai arsitektur MP3 pada {FILE_PATH}...")
    
    if not os.path.exists(FILE_PATH):
        print(f"[FATAL] Fail {FILE_PATH} tidak ditemukan.")
        sys.exit(1)

    with open(FILE_PATH, 'r') as file:
        content = file.read()

    # =================================================================
    # 1. 2-RDM Gamma Scale Factors
    # Merubah skala 1.0 menjadi 0.5 pada akumulasi 2-RDM (Gamma)
    # =================================================================
    content = re.sub(r'compute_gamma_res\(\s*t_T2t,\s*t_Taa,\s*1\.0\s*\);', 'compute_gamma_res(t_T2t, t_Taa, 0.5);', content)
    content = re.sub(r'compute_gamma_aa\(\s*t_Taa,\s*t_Taa,\s*1\.0\s*\);', 'compute_gamma_aa(t_Taa, t_Taa, 0.5);', content)
    content = re.sub(r'compute_gamma_bb\(\s*t_Tbb,\s*t_Tbb,\s*1\.0\s*\);', 'compute_gamma_bb(t_Tbb, t_Tbb, 0.5);', content)
    content = re.sub(r'compute_gamma_ab\(\s*t_Taa,\s*t_Tbb,\s*t_Tab,\s*t_Taa,\s*t_Tbb,\s*t_Tab,\s*1\.0\s*\);', 'compute_gamma_ab(t_Taa, t_Tbb, t_Tab, t_Taa, t_Tbb, t_Tab, 0.5);', content)

    # =================================================================
    # 2. Teff Accumulation Scaling
    # Merubah skala 1.0 * t2_dir menjadi 0.5 * t2_dir pada Matriks Teff
    # =================================================================
    content = re.sub(r'\+\s*1\.0\s*\*\s*\(\s*2\.0\s*\*\s*t2_dir\s*-\s*1\.0\s*\*\s*t2_ex\s*\)', '+ 0.5 * (2.0 * t2_dir - 1.0 * t2_ex)', content)
    content = re.sub(r'Teff_aa\(i\*va_\+a,\s*j\*va_\+b\)\s*=\s*1\.0\s*\*\s*t1_dir\s*\+\s*1\.0\s*\*\s*t2_dir;', 'Teff_aa(i*va_+a, j*va_+b) = 1.0 * t1_dir + 0.5 * t2_dir;', content)
    content = re.sub(r'Teff_ab\(i\*va_\+a,\s*j\*vb_\+b\)\s*=\s*1\.0\s*\*\s*t1_dir\s*\+\s*1\.0\s*\*\s*t2_dir;', 'Teff_ab(i*va_+a, j*vb_+b) = 1.0 * t1_dir + 0.5 * t2_dir;', content)
    content = re.sub(r'Teff_bb\(i\*vb_\+a,\s*j\*vb_\+b\)\s*=\s*1\.0\s*\*\s*t1_dir\s*\+\s*1\.0\s*\*\s*t2_dir;', 'Teff_bb(i*vb_+a, j*vb_+b) = 1.0 * t1_dir + 0.5 * t2_dir;', content)

    # =================================================================
    # 3. 1-RDM (OPDM Alpha)
    # Menurunkan skala T1*T2 pada matriks OPDM Alpha sebesar faktor 0.5
    # =================================================================
    # Restricted
    content = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_T2,\s*"ikab",\s*t_L2t', 'tblis::mult<double>(-0.5, t_T2, "ikab", t_L2t', content)
    content = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_L2,\s*"ikab",\s*t_T2t', 'tblis::mult<double>(-0.5, t_L2, "ikab", t_T2t', content)
    content = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_T2,\s*"ijac",\s*t_L2t', 'tblis::mult<double>(0.5, t_T2, "ijac", t_L2t', content)
    content = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_L2,\s*"ijac",\s*t_T2t', 'tblis::mult<double>(0.5, t_L2, "ijac", t_T2t', content)

    # Unrestricted AA
    content = re.sub(r'tblis::mult<double>\(\s*-0\.5,\s*t_T2aa,\s*"ikab",\s*t_T3aa', 'tblis::mult<double>(-0.25, t_T2aa, "ikab", t_T3aa', content)
    content = re.sub(r'tblis::mult<double>\(\s*-0\.5,\s*t_T3aa,\s*"ikab",\s*t_T2aa', 'tblis::mult<double>(-0.25, t_T3aa, "ikab", t_T2aa', content)
    content = re.sub(r'tblis::mult<double>\(\s*0\.5,\s*t_T2aa,\s*"ijac",\s*t_T3aa', 'tblis::mult<double>(0.25, t_T2aa, "ijac", t_T3aa', content)
    content = re.sub(r'tblis::mult<double>\(\s*0\.5,\s*t_T3aa,\s*"ijac",\s*t_T2aa', 'tblis::mult<double>(0.25, t_T3aa, "ijac", t_T2aa', content)

    # Unrestricted AB
    content = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_T2ab,\s*"ikab",\s*t_T3ab', 'tblis::mult<double>(-0.5, t_T2ab, "ikab", t_T3ab', content)
    content = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_T3ab,\s*"ikab",\s*t_T2ab', 'tblis::mult<double>(-0.5, t_T3ab, "ikab", t_T2ab', content)
    content = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_T2ab,\s*"ijac",\s*t_T3ab', 'tblis::mult<double>(0.5, t_T2ab, "ijac", t_T3ab', content)
    content = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_T3ab,\s*"ijac",\s*t_T2ab', 'tblis::mult<double>(0.5, t_T3ab, "ijac", t_T2ab', content)

    # =================================================================
    # 4. 1-RDM (OPDM Beta)
    # Menurunkan skala T1*T2 pada matriks OPDM Beta sebesar faktor 0.5
    # =================================================================
    # Unrestricted BB
    content = re.sub(r'tblis::mult<double>\(\s*-0\.5,\s*t_T2bb,\s*"ikab",\s*t_T3bb', 'tblis::mult<double>(-0.25, t_T2bb, "ikab", t_T3bb', content)
    content = re.sub(r'tblis::mult<double>\(\s*-0\.5,\s*t_T3bb,\s*"ikab",\s*t_T2bb', 'tblis::mult<double>(-0.25, t_T3bb, "ikab", t_T2bb', content)
    content = re.sub(r'tblis::mult<double>\(\s*0\.5,\s*t_T2bb,\s*"ijac",\s*t_T3bb', 'tblis::mult<double>(0.25, t_T2bb, "ijac", t_T3bb', content)
    content = re.sub(r'tblis::mult<double>\(\s*0\.5,\s*t_T3bb,\s*"ijac",\s*t_T2bb', 'tblis::mult<double>(0.25, t_T3bb, "ijac", t_T2bb', content)

    # Unrestricted AB
    content = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_T2ab,\s*"kiab",\s*t_T3ab', 'tblis::mult<double>(-0.5, t_T2ab, "kiab", t_T3ab', content)
    content = re.sub(r'tblis::mult<double>\(\s*-1\.0,\s*t_T3ab,\s*"kiab",\s*t_T2ab', 'tblis::mult<double>(-0.5, t_T3ab, "kiab", t_T2ab', content)
    content = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_T2ab,\s*"ijca",\s*t_T3ab', 'tblis::mult<double>(0.5, t_T2ab, "ijca", t_T3ab', content)
    content = re.sub(r'tblis::mult<double>\(\s*1\.0,\s*t_T3ab,\s*"ijca",\s*t_T2ab', 'tblis::mult<double>(0.5, t_T3ab, "ijca", t_T2ab', content)

    with open(FILE_PATH, 'w') as file:
        file.write(content)
    
    print("[METRIK] Operasi substitusi skalar Lagrangian OMP3 sukses dieksekusi.")

if __name__ == "__main__":
    patch_mp3_scaling()