#!/usr/bin/env python3
"""
Verifikasi gradien nuklir analitik DF-RHF, DF-UHF, dan DF-OMP2
terhadap Central Finite-Difference (fd_step = 5e-4 Bohr) dan invarian translasi.
"""
import numpy as np
import mshqc

HPCGeometryOptimizer = (
    getattr(mshqc, "HPCGeometryOptimizer", None)
    or getattr(getattr(mshqc, "_mshqc", None), "HPCGeometryOptimizer", None)
    or getattr(getattr(mshqc, "geom_opt", None), "HPCGeometryOptimizer")
)
def check_system(name: str, mol: mshqc.Molecule, method: str, basis: str = "cc-pVDZ") -> None:
    cfg_ana = mshqc.GeomOptConfig()
    cfg_ana.method = method
    cfg_ana.eri_method = "df"
    cfg_ana.basis_name = basis
    cfg_ana.use_analytic_gradient = True
    cfg_ana.project_trans_rot = False
    cfg_ana.energy_thresh = 1e-12
    cfg_ana.orb_grad_thresh = 1e-10

    cfg_fd = mshqc.GeomOptConfig()
    cfg_fd.method = method
    cfg_fd.eri_method = "df"
    cfg_fd.basis_name = basis
    cfg_fd.use_analytic_gradient = False
    cfg_fd.project_trans_rot = False
    cfg_fd.fd_step_bohr = 5e-4
    cfg_fd.energy_thresh = 1e-12
    cfg_fd.orb_grad_thresh = 1e-10

    e_ana, g_ana = mshqc.HPCGeometryOptimizer.compute_nuclear_gradient_cpp(mol, cfg_ana)
    e_fd, g_fd = mshqc.HPCGeometryOptimizer.compute_nuclear_gradient_cpp(mol, cfg_fd)

    g_ana_mat = np.asarray(g_ana).reshape(-1, 3)
    g_fd_mat = np.asarray(g_fd).reshape(-1, 3)
    max_diff = float(np.max(np.abs(g_ana_mat - g_fd_mat)))
    trans_sum = float(np.max(np.abs(np.sum(g_ana_mat, axis=0))))

    status = "PASS" if (max_diff < 1e-6 and trans_sum < 1e-11) else "FAIL"
    print(
        f"[{status}] {name:8s} | {method:7s} | E={e_ana:14.10f} | "
        f"max|g_ana - g_FD|={max_diff:.3e} | sum_trans={trans_sum:.3e}"
    )
    assert max_diff < 1e-6, f"Deviasi gradien terlalu besar pada {name} ({method}): {max_diff}"
    assert trans_sum < 1e-11, f"Invarian translasi dilanggar pada {name} ({method}): {trans_sum}"


def main() -> None:
    # 1. H2O (kulit tertutup C2v, geometri non-ekuilibrium agar gradien != 0)
    h2o = mshqc.Molecule(0, 1)
    h2o.add_atom(8, 0.0, 0.0, 0.22)
    h2o.add_atom(1, 0.0, 1.45, -0.88)
    h2o.add_atom(1, 0.0, -1.45, -0.88)

    # 2. OH radikal (kulit terbuka dublet, panjang ikatan non-ekuilibrium)
    oh = mshqc.Molecule(0, 2)
    oh.add_atom(8, 0.0, 0.0, 0.0)
    oh.add_atom(1, 0.0, 0.0, 1.90)

    # 3. NH2 radikal (poliatomik kulit terbuka dublet)
    nh2 = mshqc.Molecule(0, 2)
    nh2.add_atom(7, 0.0, 0.0, 0.20)
    nh2.add_atom(1, 0.0, 1.55, -0.85)
    nh2.add_atom(1, 0.0, -1.55, -0.85)

    for method in ("df-rhf", "df-omp2"):
        check_system("H2O", h2o, method)

    for name, mol in (("OH", oh), ("NH2", nh2)):
        for method in ("df-uhf", "df-omp2"):
            check_system(name, mol, method)


if __name__ == "__main__":
    main()
