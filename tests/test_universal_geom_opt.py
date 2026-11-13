#!/usr/bin/env python3
"""
Unit Test Standar JOSS: Verifikasi UniversalGeomOptimizer untuk
seluruh keluarga metode (RHF, UHF, ROHF, RMP2, UMP2, ROMP2, RMP3, UMP3, ROMP3,
RHF-OMP2, UHF-OMP2, RHF-OMP3, UHF-OMP3, dan MP3@OMP2).
"""

import numpy as np
from mshqc.calculators import UniversalGeomOptimizer
from mshqc.geom_opt import project_translation_rotation


def test_tr_projection_invariance():
    """Memastikan proyeksi Eckart menghilangkan gradien translasi dan rotasi murni."""
    coords = np.array([
        [0.0,  0.000000,  0.221700],
        [0.0,  1.430900, -0.886800],
        [0.0, -1.430900, -0.886800],
    ], dtype=np.float64)

    trans_grad = np.tile([0.15, -0.22, 0.08], 3)
    omega = np.array([0.05, -0.03, 0.11])
    com = np.mean(coords, axis=0)
    rot_grad = np.cross(coords - com, omega).reshape(-1)

    _ = UniversalGeomOptimizer(
        energy_fn=lambda c: 0.0,
        method="omp3",
    )
    projected = project_translation_rotation(coords, trans_grad + rot_grad)
    assert np.linalg.norm(projected) < 1e-12, f"Galat proyeksi TR: {np.linalg.norm(projected)}"
    print("[PASS] Invariansi Translasi-Rotasi (Eckart Projection) terverifikasi.")


def test_supported_methods_convergence_on_model_pes():
    """
    Menguji konvergensi UniversalGeomOptimizer untuk seluruh 14 metode yang didukung
    (dan memastikan ROHF-OMP2 / ROHF-OMP3 ditolak secara eksplisit sesuai desain).
    """
    r_eq = 1.8088
    theta_eq = np.deg2rad(104.52)

    def pes_h2o(coords: np.ndarray, e_min: float = -76.2415) -> float:
        r1 = np.linalg.norm(coords[1] - coords[0])
        r2 = np.linalg.norm(coords[2] - coords[0])
        v1 = (coords[1] - coords[0]) / r1
        v2 = (coords[2] - coords[0]) / r2
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return float(e_min + 0.45 * ((r1 - r_eq) ** 2 + (r2 - r_eq) ** 2) + 0.18 * ((theta - theta_eq) ** 2))

    init_coords = np.array([
        [0.0,  0.0000,  0.0000],
        [0.0,  1.5200,  1.1500],
        [0.0, -1.5200,  1.1500],
    ], dtype=np.float64)

    all_methods = [
        "rhf", "uhf", "rohf",
        "rmp2", "ump2", "romp2",
        "rmp3", "ump3", "romp3",
        "omp2", "uhf-omp2",
        "omp3", "uhf-omp3",
        "mp3@omp2",
    ]

    for m in all_methods:
        opt = UniversalGeomOptimizer(
            energy_fn=pes_h2o,
            method=m,
            max_iter=40,
            grad_rms_tol=1e-6,
            grad_max_tol=3e-6,
        )
        res = opt.run(init_coords)
        r1_opt = np.linalg.norm(res.final_coords_bohr[1] - res.final_coords_bohr[0])
        r2_opt = np.linalg.norm(res.final_coords_bohr[2] - res.final_coords_bohr[0])
        assert res.converged, f"Metode {m} gagal konvergen!"
        assert abs(r1_opt - r_eq) < 1e-4 and abs(r2_opt - r_eq) < 1e-4
        print(
            f"[PASS] {res.method:<10} (Ref: {res.reference:<4}) | "
            f"Iter: {res.iterations:2d} | E_final: {res.final_energy:.8f} Ha | "
            f"Grad RMS: {res.grad_rms:.2e}"
        )

    for unsupported in ("rohf-omp2", "rohf-omp3"):
        try:
            UniversalGeomOptimizer(energy_fn=pes_h2o, method=unsupported)
            raise AssertionError(f"{unsupported} seharusnya melempar NotImplementedError")
        except NotImplementedError as err:
            print(f"[PASS] Penolakan eksplisit {unsupported.upper()}: {err}")


if __name__ == "__main__":
    test_tr_projection_invariance()
    test_supported_methods_convergence_on_model_pes()
