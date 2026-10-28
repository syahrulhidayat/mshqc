#!/usr/bin/env python3
import time
import tempfile
import psi4

psi4.set_num_threads(n_threads)
psi4.set_memory("8 GB")
psi4.core.be_quiet()

# ==============================================================================
# 2. KONFIGURASI GLOBAL PSI4 (BEST-VS-BEST APPLE-TO-APPLE DENGAN MSHQC)
# ==============================================================================
BASIS_NAME = "cc-pVTZ"
AUX_BASIS  = "cc-pVTZ-RI"
THRESHOLD  = 1e-9
CHOLESKY_THRESH = 1e-9
ANG_TO_BOHR = 1.8897261246257702

ERI_MODES = ["exact", "cholesky", "df"]

Z_TO_SYM = {1: "H", 3: "Li", 6: "C", 8: "O", 10: "Ne"}

SYSTEMS = [
    ("Neon",     [(10, 0.0, 0.0, 0.0)], 0, 1, "rhf"),
    ("H2O",      [(8,  0.0, 0.0,  0.117300 * ANG_TO_BOHR),
                  (1,  0.0,  0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
                  (1,  0.0, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR)], 0, 1, "rhf"),
    ("Lithium",  [(3,  0.0, 0.0, 0.0)], 0, 2, "uhf"),
    ("Carbon",   [(6,  0.0, 0.0, 0.0)], 0, 3, "uhf"),
    ("Oxygen",   [(8,  0.0, 0.0, 0.0)], 0, 3, "uhf"),
]

RESULTS_DB = []


def build_psi4_mol(atoms, chg, mult, eri_mode="exact"):
    """
    Membangun objek molekul Psi4 pada kekuatan maksimal:
    - Mode EXACT menggunakan simetri Abelian penuh (D2h untuk atom, C2v untuk H2O).
    - Mode DF & CHOLESKY menggunakan symmetry c1 (persyaratan modul dfocc Psi4).
    """
    lines = [f"{chg} {mult}", "units bohr"]
    if eri_mode in ["df", "cholesky"]:
        lines.append("symmetry c1")
    for z, x, y, z_c in atoms:
        lines.append(f"{Z_TO_SYM[z]} {x:.12f} {y:.12f} {z_c:.12f}")
    return psi4.geometry("\n".join(lines))


def snapshot_wfn(wfn):
    """Menyimpan salinan matriks orbital & Fock HF."""
    snap = {
        "e": float(wfn.energy()),
        "Ca": wfn.Ca().clone(),
        "Da": wfn.Da().clone(),
        "Fa": wfn.Fa().clone(),
        "ea": wfn.epsilon_a().clone(),
    }
    if wfn.Cb() is not None:
        snap["Cb"] = wfn.Cb().clone()
        snap["Db"] = wfn.Db().clone()
        snap["Fb"] = wfn.Fb().clone()
        snap["eb"] = wfn.epsilon_b().clone()
    return snap


def restore_wfn(wfn, snap):
    """Mengembalikan matriks orbital & Fock HF murni tanpa menghapus cache integral."""
    wfn.set_energy(snap["e"])
    wfn.Ca().copy(snap["Ca"])
    wfn.Da().copy(snap["Da"])
    wfn.Fa().copy(snap["Fa"])
    wfn.epsilon_a().copy(snap["ea"])
    if "Cb" in snap and wfn.Cb() is not None:
        wfn.Cb().copy(snap["Cb"])
        wfn.Db().copy(snap["Db"])
        wfn.Fb().copy(snap["Fb"])
        wfn.epsilon_b().copy(snap["eb"])


def run_psi4_bench(sys_name, atoms, chg, mult, ref, basis_name, eri_mode):
    psi4.core.clean()
    psi4.core.clean_options()

    mol = build_psi4_mol(atoms, chg, mult, eri_mode)

    if eri_mode == "exact":
        scf_type, corr_type = "pk", "conv"
    elif eri_mode == "cholesky":
        scf_type, corr_type = "cd", "cd"
    elif eri_mode == "df":
        scf_type, corr_type = "df", "df"

    opts = {
        "basis": basis_name,
        "reference": ref,
        "e_convergence": THRESHOLD,
        "d_convergence": THRESHOLD,
        "scf_type": scf_type,
        "mp2_type": corr_type,
        "mp_type": corr_type,
        "cc_type": corr_type,
        "freeze_core": "false",
        "r_convergence": 1e-6,
        "rms_mograd_convergence": 1e-5,
        "max_mograd_convergence": 1e-4,
        "maxiter": 200,
        "mo_maxiter": 50,
    }

    if eri_mode == "df":
        opts.update({
            "df_basis_scf": AUX_BASIS,
            "df_basis_mp2": AUX_BASIS,
            "df_basis_cc": AUX_BASIS,
            "df_ints_io": "save",
            "read_scf_3index": True,
        })
    elif eri_mode == "cholesky":
        opts.update({
            "cholesky_tolerance": CHOLESKY_THRESH,
            "df_ints_io": "save",
            "read_scf_3index": False,
        })

    psi4.set_options(opts)

    # --- 1. SCF ---
    t0_scf = time.perf_counter()
    try:
        e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        t_scf = time.perf_counter() - t0_scf
    except Exception as e:
        print(f"  [!] SCF gagal untuk {sys_name} ({eri_mode.upper()}): {e}", flush=True)
        psi4.core.clean()
        return None, 0.0, None, None, 0.0, None, 0.0, None, 0.0, None, 0.0

    hf_snap = snapshot_wfn(scf_wfn)

    # Untuk CHOLESKY, matikan read_scf_3index namun simpan integral CD di Unit 97 (df_ints_io=load)
    if eri_mode == "cholesky":
        psi4.set_options({"scf_type": "pk", "read_scf_3index": False})

    # --- 2. MP2 ---
    mp2_label = "RMP2" if ref == "rhf" else "UMP2"
    t0_mp2 = time.perf_counter()
    try:
        e_mp2 = psi4.energy("mp2", ref_wfn=scf_wfn, molecule=mol)
        t_mp2 = time.perf_counter() - t0_mp2
    except Exception:
        e_mp2, t_mp2 = None, 0.0
    if eri_mode != "df":
        psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # --- 3. MP3 ---
    mp3_label = "RMP3" if ref == "rhf" else "UMP3"
    t0_mp3 = time.perf_counter()
    try:
        e_mp3 = psi4.energy("mp3", ref_wfn=scf_wfn, molecule=mol)
        t_mp3 = time.perf_counter() - t0_mp3
    except Exception:
        e_mp3, t_mp3 = None, 0.0
    if eri_mode != "df":
        psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # --- 4. OMP2 ---
    t0_omp2 = time.perf_counter()
    try:
        e_omp2 = psi4.energy("omp2", ref_wfn=scf_wfn, molecule=mol)
        t_omp2 = time.perf_counter() - t0_omp2
    except Exception:
        e_omp2, t_omp2 = None, 0.0
    if eri_mode != "df":
        psi4.core.clean()
    restore_wfn(scf_wfn, hf_snap)

    # --- 5. OMP3 ---
    t0_omp3 = time.perf_counter()
    try:
        e_omp3 = psi4.energy("omp3", ref_wfn=scf_wfn, molecule=mol)
        t_omp3 = time.perf_counter() - t0_omp3
    except Exception:
        e_omp3, t_omp3 = None, 0.0
    psi4.core.clean()

    return e_scf, t_scf, mp2_label, e_mp2, t_mp2, mp3_label, e_mp3, t_mp3, e_omp2, t_omp2, e_omp3, t_omp3


def main():
    print("==========================================================================================================", flush=True)
    print("  PSI4 BENCHMARK (MAX PERFORMANCE) | EXACT vs CHOLESKY vs DF (HF, MP2, MP3, OMP2, OMP3)", flush=True)
    print(f"  Basis: {BASIS_NAME} | Aux Basis: {AUX_BASIS} | Chol Tol: {CHOLESKY_THRESH} | E_conv: 1e-09 | Grad_conv: 1e-05", flush=True)
    print("==========================================================================================================", flush=True)

    for sys_name, atoms, chg, mult, ref in SYSTEMS:
        ref_label = ref.upper()
        mp2_lbl = "RMP2" if ref == "rhf" else "UMP2"
        mp3_lbl = "RMP3" if ref == "rhf" else "UMP3"

        for eri_mode in ERI_MODES:
            print(f"Running: {sys_name:<8} -> {ref_label}, {mp2_lbl}, {mp3_lbl}, OMP2, & OMP3 (Integral: {eri_mode.upper()})", flush=True)

            res = run_psi4_bench(sys_name, atoms, chg, mult, ref, BASIS_NAME, eri_mode)
            e_scf, t_scf, mp2_label, e_mp2, t_mp2, mp3_label, e_mp3, t_mp3, e_omp2, t_omp2, e_omp3, t_omp3 = res

            if e_scf is not None: RESULTS_DB.append({"sys": sys_name, "eri": eri_mode.upper(), "method": ref_label, "psi4_e": e_scf, "psi4_t": t_scf})
            if e_mp2 is not None: RESULTS_DB.append({"sys": sys_name, "eri": eri_mode.upper(), "method": mp2_label, "psi4_e": e_mp2, "psi4_t": t_mp2})
            if e_mp3 is not None: RESULTS_DB.append({"sys": sys_name, "eri": eri_mode.upper(), "method": mp3_label, "psi4_e": e_mp3, "psi4_t": t_mp3})
            if e_omp2 is not None: RESULTS_DB.append({"sys": sys_name, "eri": eri_mode.upper(), "method": "OMP2", "psi4_e": e_omp2, "psi4_t": t_omp2})
            if e_omp3 is not None: RESULTS_DB.append({"sys": sys_name, "eri": eri_mode.upper(), "method": "OMP3", "psi4_e": e_omp3, "psi4_t": t_omp3})

    print("\n===============================================================================", flush=True)
    print("                        PSI4 BENCHMARK FINAL RESULTS", flush=True)
    print("===============================================================================", flush=True)
    header = f"| {'System':<8} | {'ERI Mode':<10} | {'Method':<8} | {'Energy (Ha)':<16} | {'Time (s)':<8} |"
    print("-" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for row in RESULTS_DB:
        print(f"| {row['sys']:<8} | {row['eri']:<10} | {row['method']:<8} | {row['psi4_e']:<16.8f} | {row['psi4_t']:<8.3f} |", flush=True)
    print("-" * len(header), flush=True)


if __name__ == "__main__":
    main()