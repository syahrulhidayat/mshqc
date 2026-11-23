#!/usr/bin/env python3
"""
Verifikasi Smoke Suite Benchmark v2.0:
1. Core-5 (pengganti `test_mshqc2`: Neon, H2O, Lithium, Carbon, Oxygen di cc-pVDZ DF & Cholesky).
2. Eksekusi langsung `benchmarks/run_bab4_v2.py --quick --skip-ram` (exit 0 + JSON valid).
3. Eksekusi langsung `benchmarks/run_universal_v2.py --tier quick --basis cc-pVDZ --extended` (exit 0 + 6/6 PASS).
4. Verifikasi HPC Exact 4D TBLIS (`SCF/MP2/OMP2/OMP3`, `transform_eri_4d`) & ringkasan `__str__`.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import mshqc


def test_core5_systems_df_and_cholesky():
    core5 = [
        ("Neon", [("Ne", 0.0, 0.0, 0.0)], 0, 1),
        ("H2O", [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)], 0, 1),
        ("Lithium", [("Li", 0.0, 0.0, 0.0)], 0, 2),
        ("Carbon", [("C", 0.0, 0.0, 0.0)], 0, 3),
        ("Oxygen", [("O", 0.0, 0.0, 0.0)], 0, 3),
    ]
    for name, atoms, q, mult in core5:
        for mode in ("df", "cholesky"):
            res = mshqc.run(
                atoms, method=["scf", "mp2", "mp3", "omp2", "omp3"],
                basis="cc-pVDZ", eri_method=mode, charge=q, multiplicity=mult,
                units="angstrom", threshold=1e-9, grad_thresh=1e-5,
            )
            e_hf = res["scf"]["energy"]
            e_o2 = res["omp2"]["energy"]
            e_o3 = res["omp3"]["energy"]
            assert res["scf"]["result"].converged
            assert res["omp2"]["result"].converged and res["omp3"]["result"].converged
            assert e_o2 < e_hf and np.isfinite(e_o3)
    print("[CORE-5 v2.0] Neon/H2O/Li/C/O (DF + Cholesky, SCF->MP2->MP3->OMP2->OMP3) PASS")


def test_bab4_v2_and_universal_v2_runners_execute_cleanly():
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        b4_json = td_p / "bab4_quick.json"
        b4_log = td_p / "bab4_quick.log"
        subprocess.run(
            [
                sys.executable, str(root / "benchmarks" / "run_bab4_v2.py"),
                "--quick", "--skip-ram", "--out", str(b4_json), "--log", str(b4_log),
            ],
            check=True,
            cwd=root,
        )
        b4_data = json.loads(b4_json.read_text(encoding="utf-8"))
        assert "H2O|DF" in b4_data["ref_suite"] and "OH|DF" in b4_data["ref_suite"]
        assert all(v["err"] < 1.0e-7 for v in b4_data["fd_summary"].values())

        uni_json = td_p / "uni_quick.json"
        subprocess.run(
            [
                sys.executable, str(root / "benchmarks" / "run_universal_v2.py"),
                "--tier", "quick", "--basis", "cc-pVDZ", "--extended", "--out", str(uni_json),
            ],
            check=True,
            cwd=root,
        )
        uni_data = json.loads(uni_json.read_text(encoding="utf-8"))
        assert uni_data["n_failed"] == 0 and uni_data["n_passed"] == 6
        print(
            f"[RUNNERS v2 ] bab4_v2 (--quick) OK | universal_v2 (--tier quick): "
            f"{uni_data['n_passed']}/6 PASS"
        )


def test_exact_eri_parity_and_human_str():
    h2o = "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692"
    r_ex = mshqc.run(h2o, method=["scf", "mp2", "omp2", "omp3"], basis="cc-pVDZ", eri_method="exact", units="angstrom", threshold=1e-9)
    r_cd = mshqc.run(h2o, method=["scf", "mp2", "omp2", "omp3"], basis="cc-pVDZ", eri_method="cholesky", units="angstrom", threshold=1e-9)
    for m in ("scf", "mp2", "omp2", "omp3"):
        assert abs(r_ex[m]["energy"] - r_cd[m]["energy"]) < 1.0e-8, f"EXACT vs CHOL mismatch on {m}"
    assert abs(r_ex["omp2"]["energy"] - (-76.231670)) < 2.0e-6
    assert abs(r_ex["omp3"]["energy"] - (-76.238143)) < 2.0e-6

    mol = mshqc.build_molecule(h2o, units="angstrom")
    bas = mshqc.BasisSet("cc-pVDZ", mol)
    eri4 = mshqc.IntegralEngine(mol, bas).compute_eri()
    C = r_ex["scf"]["result"].C_alpha
    V_oovv = mshqc.kernels.transform_eri_4d(eri4, C[:, :5], C[:, 5:], C[:, :5], C[:, 5:])
    assert V_oovv.shape == (5 * 19, 5 * 19)

    s_scf = str(r_ex["scf"]["result"])
    s_omp = str(r_ex["omp2"]["result"])
    assert len(s_scf) < 160 and "E_total" in s_scf
    assert len(s_omp) < 200 and "OMP2" in s_omp
    oh = "O 0 0 0.0\nH 0 0 0.9697"
    u_ex = mshqc.run(oh, method=["scf", "mp3", "omp2.5", "omp3"], basis="cc-pVDZ", eri_method="exact", units="angstrom", threshold=1e-9)
    u_cd = mshqc.run(oh, method=["scf", "mp3", "omp2.5", "omp3"], basis="cc-pVDZ", eri_method="cholesky", units="angstrom", threshold=1e-9)
    for m in ("scf", "mp3", "omp2.5", "omp3"):
        assert abs(u_ex[m]["energy"] - u_cd[m]["energy"]) < 1.0e-8, f"EXACT vs CHOL UHF mismatch on {m}"
    print(f"[EXACT 4D  ] {s_omp} | OH OMP3={u_ex['omp3']['energy']:.8f} Ha")


if __name__ == "__main__":
    test_core5_systems_df_and_cholesky()
    test_bab4_v2_and_universal_v2_runners_execute_cleanly()
    test_exact_eri_parity_and_human_str()
    print("[PASS] Semua validasi smoke benchmark v2.0 lulus!")
