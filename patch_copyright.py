import os
import sys

def apply_omp3_scalar_fixes(filepath="src/mp3/mp3.cc"):
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan!")
        sys.exit(1)

    with open(filepath, 'r') as file:
        content = file.read()

    # =========================================================================
    # 1. KOREKSI SISTEM RESTRICTED (Faktor 0.5x)
    # =========================================================================
    
    # OVOV: 2.0 -> 1.0
    content = content.replace(
        'tblis::mult<double>(-2.0, t_Y_aa, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(-1.0, t_Y_aa, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>( 2.0, t_Y_aa, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");',
        'tblis::mult<double>( 1.0, t_Y_aa, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");'
    )
    
    # VVVV: 1.0 -> 0.5
    content = content.replace(
        'tblis::mult<double>(1.0, t_Yvv_a, "acP", t_Bia_a, "ciP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(0.5, t_Yvv_a, "acP", t_Bia_a, "ciP", 1.0, t_Za, "ai");'
    )
    
    # OOOO: -1.0 -> -0.5
    content = content.replace(
        'tblis::mult<double>(-1.0, t_Yoo_a, "ikP", t_Bia_a, "akP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(-0.5, t_Yoo_a, "ikP", t_Bia_a, "akP", 1.0, t_Za, "ai");'
    )


    # =========================================================================
    # 2. KOREKSI SISTEM UNRESTRICTED (Faktor 0.25x)
    # =========================================================================

    # OVOV: 4.0 -> 1.0
    content = content.replace(
        'tblis::mult<double>(-4.0, t_Y_aa, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(-1.0, t_Y_aa, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>( 4.0, t_Y_aa, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");',
        'tblis::mult<double>( 1.0, t_Y_aa, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>(-4.0, t_Y_bb, "amP", t_Boo_b, "miP", 1.0, t_Zb, "bi");',
        'tblis::mult<double>(-1.0, t_Y_bb, "amP", t_Boo_b, "miP", 1.0, t_Zb, "bi");'
    )
    content = content.replace(
        'tblis::mult<double>( 4.0, t_Y_bb, "eiP", t_Bvv_b, "aeP", 1.0, t_Zb, "bi");',
        'tblis::mult<double>( 1.0, t_Y_bb, "eiP", t_Bvv_b, "aeP", 1.0, t_Zb, "bi");'
    )
    content = content.replace(
        'tblis::mult<double>(-4.0, t_Y_ab_a, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(-1.0, t_Y_ab_a, "amP", t_Boo_a, "miP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>( 4.0, t_Y_ab_a, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");',
        'tblis::mult<double>( 1.0, t_Y_ab_a, "eiP", t_Bvv_a, "aeP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>(-4.0, t_Y_ab_b, "bmP", t_Boo_b, "mjP", 1.0, t_Zb, "bj");',
        'tblis::mult<double>(-1.0, t_Y_ab_b, "bmP", t_Boo_b, "mjP", 1.0, t_Zb, "bj");'
    )
    content = content.replace(
        'tblis::mult<double>( 4.0, t_Y_ab_b, "ejP", t_Bvv_b, "beP", 1.0, t_Zb, "bj");',
        'tblis::mult<double>( 1.0, t_Y_ab_b, "ejP", t_Bvv_b, "beP", 1.0, t_Zb, "bj");'
    )

    # VVVV: 2.0 -> 0.5
    content = content.replace(
        'tblis::mult<double>(2.0, t_Yvv_a, "acP", t_Bia_a, "ciP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(0.5, t_Yvv_a, "acP", t_Bia_a, "ciP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>(2.0, t_Yvv_b, "bdP", t_Bia_b, "diP", 1.0, t_Zb, "bi"); ',
        'tblis::mult<double>(0.5, t_Yvv_b, "bdP", t_Bia_b, "diP", 1.0, t_Zb, "bi"); '
    )

    # OOOO: -2.0 -> -0.5
    content = content.replace(
        'tblis::mult<double>(-2.0, t_Yoo_a, "ikP", t_Bia_a, "akP", 1.0, t_Za, "ai");',
        'tblis::mult<double>(-0.5, t_Yoo_a, "ikP", t_Bia_a, "akP", 1.0, t_Za, "ai");'
    )
    content = content.replace(
        'tblis::mult<double>(-2.0, t_Yoo_b, "lnP", t_Bia_b, "alP", 1.0, t_Zb, "an"); ',
        'tblis::mult<double>(-0.5, t_Yoo_b, "lnP", t_Bia_b, "alP", 1.0, t_Zb, "an"); '
    )

    with open(filepath, 'w') as file:
        file.write(content)
        
    print("[+] Normalisasi skalar Z-Vector OMP3 berhasil diterapkan!")

if __name__ == "__main__":
    apply_omp3_scalar_fixes()