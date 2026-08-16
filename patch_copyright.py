import re
import os
import sys

def apply_omp3_gamma_theory(filepath="src/mp3/mp3.cc"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan!")
        sys.exit(1)

    with open(filepath, 'r') as file:
        content = file.read()

    # Pemetaan tensor T2 ke T3
    tensor_map = {
        "t_Taa": "t_T3aa",
        "t_Tbb": "t_T3bb",
        "t_Tab": "t_T3ab",
        "t_T2t": "t_L2t"
    }

    def process_line(match):
        val = float(match.group(1))
        t_a = match.group(2)
        str_a = match.group(3)
        t_b = match.group(4)
        str_b = match.group(5)
        beta = match.group(6)
        t_out = match.group(7)
        str_out = match.group(8)

        t3_a = tensor_map[t_a]
        t3_b = tensor_map[t_b]

        # Bobot T2*T2 digandakan (mengakomodasi derivatif eksplisit Gamma_W)
        val_w = val * 2.0
        val_d = val

        # Injeksi term T2*T3 dan T3*T2 (mengakomodasi derivatif denominator Gamma_D)
        res = f'tblis::mult<double>({val_w}, {t_a}, "{str_a}", {t_b}, "{str_b}", {beta}, {t_out}, "{str_out}");\n'
        res += f'        tblis::mult<double>({val_d}, {t_a}, "{str_a}", {t3_b}, "{str_b}", 1.0, {t_out}, "{str_out}");\n'
        res += f'        tblis::mult<double>({val_d}, {t3_a}, "{str_a}", {t_b}, "{str_b}", 1.0, {t_out}, "{str_out}");'
        return res

    # Hanya menargetkan matriks Gvvvv, Goooo, dan Govov. (Mengecualikan T_eff dan OPDM)
    pattern = r'tblis::mult<double>\(([-0-9.]+),\s*(t_T(?:aa|bb|ab|2t)),\s*"([^"]+)",\s*(t_T(?:aa|bb|ab|2t)),\s*"([^"]+)",\s*([0-9.]+),\s*(t_G(?:vvvv|oooo|ovov)[a-z_]*),\s*"([^"]+)"\);'
    
    new_content, num_subs = re.subn(pattern, process_line, content)

    if num_subs == 0:
        print("[-] Tidak ada tensor yang dimodifikasi. Pastikan kode OMP3 Anda tidak diubah strukturnya.")
    else:
        with open(filepath, 'w') as file:
            file.write(new_content)
        print(f"[+] TEORI OMP3 TBLIS SEMPURNA! {num_subs} kontraksi berhasil dimutasi untuk mengakomodasi Gamma_W dan Gamma_D.")

if __name__ == "__main__":
    apply_omp3_gamma_theory()