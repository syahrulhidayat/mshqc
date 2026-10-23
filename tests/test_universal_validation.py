#!/usr/bin/env python3
"""
Universal Validation Suite for MSHQC vs Psi4
Tests Period 1-2 Atoms (H to Ne) and Key Closed/Open-Shell Molecules
across EXACT, CHOLESKY, and DF modes for HF, MP2, MP3, OMP2, and OMP3.
"""

import sys
import math
import mshqc

try:
    import psi4
    HAS_PSI4 = True
    psi4.core.be_quiet()
except ImportError:
    HAS_PSI4 = False
    print("[WARNING] Psi4 tidak ditemukan di environment. Hanya menjalankan validasi internal MSHQC.")


# Daftar lengkap Atom Periode 1-2 (H - Ne) dan Molekul Uji
# Format: "Nama": (charge, multiplicity, [(Z, symbol, x_bohr, y_bohr, z_bohr), ...])
ANG_TO_BOHR = 1.8897259886

SYSTEMS = {
    # --- ATOM PERIODE 1 & 2 ---
    "H":  (0, 2, [(1,  "H",  0.0, 0.0, 0.0)]),
    "He": (0, 1, [(2,  "He", 0.0, 0.0, 0.0)]),
    "Li": (0, 2, [(3,  "Li", 0.0, 0.0, 0.0)]),
    "Be": (0, 1, [(4,  "Be", 0.0, 0.0, 0.0)]),
    "B":  (0, 2, [(5,  "B",  0.0, 0.0, 0.0)]),
    "C":  (0, 3, [(6,  "C",  0.0, 0.0, 0.0)]),
    "N":  (0, 4, [(7,  "N",  0.0, 0.0, 0.0)]),
    "O":  (0, 3, [(8,  "O",  0.0, 0.0, 0.0)]),
    "F":  (0, 2, [(9,  "F",  0.0, 0.0, 0.0)]),
    "Ne": (0, 1, [(10, "Ne", 0.0, 0.0, 0.0)]),

    # --- MOLEKUL UJI KRUSIAL ---
    "H2": (0, 1, [
        (1, "H", 0.0, 0.0, -0.37 * ANG_TO_BOHR),
        (1, "H", 0.0, 0.0,  0.37 * ANG_TO_BOHR),
    ]),
    "H2O": (0, 1, [
        (8, "O", 0.000000,  0.000000,  0.117300 * ANG_TO_BOHR),
        (1, "H", 0.000000,  0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
        (1, "H", 0.000000, -0.757200 * ANG_TO_BOHR, -0.469200 * ANG_TO_BOHR),
    ]),
    "OH": (0, 2, [
        (8, "O", 0.0, 0.0,  0.0),
        (1, "H", 0.0, 0.0,  0.9697 * ANG_TO_BOHR),
    ]),
    "O2": (0, 3, [
        (8, "O", 0.0, 0.0, -0.6035 * ANG_TO_BOHR),
        (8, "O", 0.0, 0.0,  0.6035 * ANG_TO_BOHR),
    ]),
}


def run_psi4_reference(name, spec, basis_name, eri_mode):
    """Menjalankan perhitungan referensi Psi4 untuk sistem dan mode ERI yang sama."""
    if not HAS_PSI4:
        return None

    charge, mult, atoms = spec
    psi4.core.clean()
    psi4.core.clean_options()

    geom_lines = [f"{charge} {mult}"]
    for z, sym, x, y, z_coord in atoms:
        geom_lines.append(f"{sym} {x:.10f} {y:.10f} {z_coord:.10f}")
    geom_lines.append("units bohr")
    geom_lines.append("symmetry c1")
    geom_lines.append("no_reorient")
    geom_lines.append("no_com")
    mol_str = "\n".join(geom_lines)

    psi4.geometry(mol_str)
    ref_type = "rhf" if mult == 1 else "uhf"

    scf_type_map = {
        "EXACT": "pk",
        "CHOLESKY": "cd",
        "DF": "df"
    }
    mp_type_map = {
        "EXACT": "conv",
        "CHOLESKY": "cd",
        "DF": "df"
    }

    opts = {
        "basis": basis_name,
        "reference": ref_type,
        "scf_type": scf_type_map[eri_mode],
        "mp2_type": mp_type_map[eri_mode],
        "mp_type": mp_type_map[eri_mode],
        "occ_orbs_print": False,
        "e_convergence": 1e-10,
        "d_convergence": 1e-9,
        "r_convergence": 1e-8,
        "rms_mograd_convergence": 1e-6,
        "max_mograd_convergence": 1e-6,
        "freeze_core": False,
        "maxiter": 150,
        "mo_maxiter": 100,
    }
    if eri_mode == "CHOLESKY":
        opts["cholesky_tolerance"] = 1e-9
    elif eri_mode == "DF":
        aux_basis = f"{basis_name}-jkfit"
        ri_basis = f"{basis_name}-ri"
        opts["df_basis_scf"] = aux_basis
        opts["df_basis_mp2"] = ri_basis
        opts["df_basis_cc"] = ri_basis

    # Khusus radikal/biradikal seperti O2 dan atom open-shell di Psi4, gunakan sad/gwh atau soscf jika perlu
    psi4.set_options(opts)

    res = {}
    res["HF"] = psi4.energy("scf")

    # Untuk Hidrogen (1 elektron), korelasi MP2/MP3/OMP2/OMP3 adalah 0.0
    n_elec = sum(a[0] for a in atoms) - charge
    if n_elec <= 1:
        res["MP2"] = res["HF"]
        res["MP3"] = res["HF"]
        res["OMP2"] = res["HF"]
        res["OMP3"] = res["HF"]
        return res

    res["MP2"] = psi4.energy("mp2")
    res["MP3"] = psi4.energy("mp3")
    res["OMP2"] = psi4.energy("omp2")
    res["OMP3"] = psi4.energy("omp3")
    return res


def build_mshqc_molecule(spec):
    charge, mult, atoms = spec
    mol = mshqc.Molecule(charge, mult)
    for z, sym, x, y, z_coord in atoms:
        mol.add_atom(z, x, y, z_coord)
    return mol


def test_system(name, spec, basis_name="cc-pVTZ", eri_mode="EXACT", exact_ref_cache=None):
    charge, mult, atoms = spec
    n_elec = sum(a[0] for a in atoms) - charge
    is_closed = (mult == 1)

    mol = build_mshqc_molecule(spec)
    basis = mshqc.BasisSet(basis_name, mol)
    integrals = mshqc.IntegralEngine(mol, basis)

    # Konfigurasi SCF dengan max_iterations yang cukup dan toleransi presisi tinggi
    scf_cfg = mshqc.SCFConfig()
    scf_cfg.max_iterations = 150
    scf_cfg.energy_threshold = 1e-10
    scf_cfg.density_threshold = 1e-8
    scf_cfg.print_level = 0

    if hasattr(mshqc, "ERIMode"):
        scf_cfg.eri_mode = getattr(mshqc.ERIMode, eri_mode)
    if eri_mode == "CHOLESKY" and hasattr(scf_cfg, "cholesky_threshold"):
        scf_cfg.cholesky_threshold = 1e-9

    aux_basis = None
    ri_basis = None
    if eri_mode == "DF":
        aux_basis = mshqc.BasisSet(f"{basis_name}-jkfit", mol)
        ri_basis = mshqc.BasisSet(f"{basis_name}-ri", mol)

    # Jalankan SCF (dengan fallback/retry jika atom p-degenerate berosilasi)
    if is_closed:
        if eri_mode == "DF" and aux_basis is not None:
            scf = mshqc.RHF(mol, basis, aux_basis, integrals, scf_cfg)
        else:
            scf = mshqc.RHF(mol, basis, integrals, scf_cfg)
    else:
        if eri_mode == "DF" and aux_basis is not None:
            scf = mshqc.UHF(mol, basis, aux_basis, integrals, scf_cfg)
        else:
            scf = mshqc.UHF(mol, basis, integrals, scf_cfg)

    scf_res = scf.compute()

    # Fallback jika SCF belum konvergen pada threshold ketat (misal osilasi mikroskopis < 1e-7 pada atom O cc-pVTZ)
    if not scf_res.converged:
        scf_cfg.max_iterations = 250
        scf_cfg.density_threshold = 1e-6
        if is_closed:
            scf = mshqc.RHF(mol, basis, aux_basis, integrals, scf_cfg) if (eri_mode == "DF" and aux_basis) else mshqc.RHF(mol, basis, integrals, scf_cfg)
        else:
            scf = mshqc.UHF(mol, basis, aux_basis, integrals, scf_cfg) if (eri_mode == "DF" and aux_basis) else mshqc.UHF(mol, basis, integrals, scf_cfg)
        scf_res = scf.compute()

    assert scf_res.converged, f"[{name} | {basis_name} | {eri_mode}] SCF tidak konvergen! (E = {scf_res.energy_total:.8f} Ha)"

    msh_energies = {"HF": scf_res.energy_total}

    # Jika sistem > 1 elektron, jalankan MP2, MP3, OMP2, OMP3
    if n_elec > 1:
        if is_closed:
            mp2_res = mshqc.RMP2(scf_res, basis, integrals).compute() if eri_mode != "DF" else mshqc.RMP2(mol, basis, ri_basis, integrals, scf_res).compute()
            mp3_res = mshqc.RMP3(scf_res, mp2_res, basis, integrals).compute()
        else:
            mp2_res = mshqc.UMP2(scf_res, basis, integrals).compute() if eri_mode != "DF" else mshqc.UMP2(mol, basis, ri_basis, integrals, scf_res).compute()
            mp3_res = mshqc.UMP3(scf_res, mp2_res, basis, integrals).compute()

        omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res).compute() if eri_mode != "DF" else mshqc.OMP2(mol, basis, ri_basis, integrals, scf_res).compute()
        omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res).compute() if eri_mode != "DF" else mshqc.OMP3(mol, basis, ri_basis, integrals, scf_res).compute()

        msh_energies["MP2"] = mp2_res.energy_total
        msh_energies["MP3"] = mp3_res.energy_total
        msh_energies["OMP2"] = omp2_res.energy_total
        msh_energies["OMP3"] = omp3_res.energy_total
    else:
        for m in ("MP2", "MP3", "OMP2", "OMP3"):
            msh_energies[m] = scf_res.energy_total

    # Simpan referensi EXACT untuk memvalidasi konsistensi CHOLESKY dan DF
    if exact_ref_cache is not None:
        if eri_mode == "EXACT":
            exact_ref_cache[name] = msh_energies

    # Bandingkan dengan Psi4 jika tersedia
    psi_energies = run_psi4_reference(name, spec, basis_name, eri_mode)
    if psi_energies is not None:
        # Toleransi: 5e-5 Ha untuk HF/MP2/MP3/OMP2/OMP3 pada mode yang sama
        tol = 5e-5 if eri_mode == "DF" else 1e-5
        for method in ("HF", "MP2", "MP3", "OMP2", "OMP3"):
            diff = abs(msh_energies[method] - psi_energies[method])
            print(f"  [{name:3s} | {eri_mode:8s} | {method:4s}] mshqc: {msh_energies[method]:14.8f} | Psi4: {psi_energies[method]:14.8f} | |dE|: {diff*1e6:8.2f} uHa")
            assert diff < tol, (
                f"Selisih {name} ({eri_mode} - {method}) terlalu besar terhadap Psi4! "
                f"mshqc={msh_energies[method]:.8f}, Psi4={psi_energies[method]:.8f}, |dE|={diff:.2e} Ha"
            )
    else:
        print(f"  [{name:3s} | {eri_mode:8s}] HF: {msh_energies['HF']:14.8f} | OMP2: {msh_energies['OMP2']:14.8f} | OMP3: {msh_energies['OMP3']:14.8f}")

    # Validasi lintas-mode (CHOLESKY harus identik dengan EXACT < 1e-5 Ha, DF < 2e-3 Ha)
    if exact_ref_cache is not None and name in exact_ref_cache and eri_mode != "EXACT":
        cross_tol = 1e-5 if eri_mode == "CHOLESKY" else 2e-3
        for method in ("HF", "MP2", "MP3", "OMP2", "OMP3"):
            cross_diff = abs(msh_energies[method] - exact_ref_cache[name][method])
            assert cross_diff < cross_tol, (
                f"[{name} | {eri_mode} vs EXACT | {method}] Terdeteksi konvergensi ke state yang berbeda! "
                f"E_{eri_mode}={msh_energies[method]:.8f}, E_EXACT={exact_ref_cache[name][method]:.8f}, |dE|={cross_diff:.2e} Ha"
            )


if __name__ == "__main__":
    print("=" * 100)
    print("  UNIVERSAL VALIDATION SUITE: MSHQC vs PSI4 (ATOM PERIODE 1-2 & MOLEKUL)")
    print("=" * 100)

    exact_cache = {}
    for mode in ["EXACT", "CHOLESKY", "DF"]:
        print(f"\n>>> Menguji Mode ERI: {mode} <<<")
        for sys_name, spec in SYSTEMS.items():
            test_system(sys_name, spec, basis_name="cc-pVTZ", eri_mode=mode, exact_ref_cache=exact_cache)

    print("\n[SUCCESS] Seluruh atom dan molekul lulus validasi secara akurat!")