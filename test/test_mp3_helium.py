import mshqc as mshqc
import sys
import math

def test_helium_rmp2_rmp3():
    # 1. Inisialisasi Sistem Molekuler (Helium, singlet, netral)
    mol = mshqc.Molecule(0, 1) # charge = 0, multiplicity = 1
    mol.add_atom(2, 0.0, 0.0, 0.0)

    # 2. Inisialisasi Basis Set
    # Gunakan basis set standar yang mendukung polarisasi untuk resolusi korelasi elektron
    basis_name = "cc-pVTZ"
    try:
        basis = mshqc.BasisSet(basis_name, mol, "data/basis")
    except Exception as e:
        print(f"[FATAL] Gagal memuat basis set {basis_name}. Pastikan direktori data/basis tersedia.")
        sys.exit(1)

    # 3. Alokasi Engine Integral
    integrals = mshqc.IntegralEngine(mol, basis)
    
    # 4. Eksekusi Reference State (Mean-Field RHF)
    scf_conf = mshqc.SCFConfig()
    scf_conf.energy_threshold = 1e-10
    scf_conf.density_threshold = 1e-10
    scf_conf.max_iterations = 100
    
    rhf_engine = mshqc.RHF(mol, basis, integrals, scf_conf)
    print("Mengeksekusi SCF (RHF)...")
    scf_res = rhf_engine.compute()
    
    if not scf_res.converged:
        print("[ERROR] RHF gagal konvergen. Terminasi proses.")
        sys.exit(1)
        
    print(f"Energi SCF (HF): {scf_res.energy_total:.8f} a.u.")

    # 5. Eksekusi RMP2 (Tensor Contraction Orde 2)
    mp2_conf = mshqc.MP2Config()
    mp2_conf.scf_type = "RHF" # FFI nanobind mensyaratkan std::string
    
    rmp2_engine = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf)
    print("Mengeksekusi RMP2...")
    mp2_res = rmp2_engine.compute()
    
    print(f"Energi Korelasi MP2: {mp2_res.energy_mp2_corr:.8f} a.u.")
    print(f"Energi Total MP2   : {mp2_res.energy_total:.8f} a.u.")

    # 6. Eksekusi RMP3 (Tensor Contraction Orde 3)
    # Memerlukan SCF state, MP2 state (amplitudo T2), MP2 config, dan reference memory Integral
    rmp3_engine = mshqc.RMP3(scf_res, mp2_res, mp2_conf, integrals)
    print("Mengeksekusi RMP3...")
    mp3_res = rmp3_engine.compute()
    
    print(f"Energi Korelasi Total MP3: {mp3_res.e_corr_total:.8f} a.u.")
    print(f"Energi Total MP3         : {mp3_res.e_total:.8f} a.u.")

    # 7. Evaluasi Empiris Limit Fisik (Grounding Test)
    # Energi eksak non-relativistik Helium berada di sekitar -2.903 a.u.
    # Jika energi total menembus batas ini secara berlebihan, terdapat de-sinkronisasi pada backend C++.
    exact_limit = -2.90338583
    tolerance = 1e-3

    if mp3_res.e_total < (exact_limit - tolerance):
        print(f"[ANOMALI TERDETEKSI] Energi MP3 ({mp3_res.e_total:.8f} a.u.) melampaui batas eksak fisik Helium ({exact_limit} a.u.).")
        print("Indikasi kuat: Kerusakan indeksasi tensor pada modul double excitation atau memory stride error pada ERI.")
        sys.exit(1)
    else:
        print("[VALIDASI] Eksekusi analitik MP2/MP3 berjalan secara stabil di dalam batas fisik yang valid.")

if __name__ == "__main__":
    test_helium_rmp2_rmp3()