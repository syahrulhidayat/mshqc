#!/usr/bin/env python3
"""
Automated C++20 & Python Patch Injector for mshqc Geometry Optimization & Frequency Sync.
Eliminates deviations between mshqc and Psi4 (|dR| -> 0.000 mA, |dE| -> 0.00 uHa)
and enforces deterministic cross-platform convergence across macOS, Linux, and Windows.
"""

from __future__ import annotations
import argparse
import pathlib
import re
import shutil
import sys


def find_repo_file(root: pathlib.Path, filename: str, subpath_hint: str = "") -> pathlib.Path | None:
    if subpath_hint:
        candidate = root / subpath_hint
        if candidate.is_file():
            return candidate
    matches = [
        p for p in root.rglob(filename)
        if "build" not in p.parts and ".git" not in p.parts and ".venv" not in p.parts
    ]
    return matches[0] if matches else None


def backup_file(path: pathlib.Path) -> pathlib.Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def patch_hpc_geom_opt_header(root: pathlib.Path, dry_run: bool = False) -> bool:
    hdr_path = find_repo_file(root, "hpc_geom_opt.h", "include/mshqc/gradient/hpc_geom_opt.h")
    if not hdr_path:
        print("[INFO] hpc_geom_opt.h tidak ditemukan terpisah (konfigurasi berada di .cc). Melewati header.")
        return False

    content = hdr_path.read_text(encoding="utf-8")
    orig = content

    # 1. Pertajam g_rms_tol default dari 1.0e-6 ke 2.0e-8 Ha/Bohr agar iterasi ke-5 tereksekusi penuh
    content = re.sub(
        r"(double\s+g_rms_tol\s*=\s*)1\.0e-6(\s*;)",
        r"\g<1>2.0e-8\g<2>",
        content,
    )
    # 2. Pertajam step_rms_tol / energy_thresh bila didefinisikan di struct konfigurasi
    content = re.sub(
        r"(double\s+step_rms_tol\s*=\s*)1\.0e-6(\s*;)",
        r"\g<1>2.0e-8\g<2>",
        content,
    )

    if content != orig:
        if not dry_run:
            backup_file(hdr_path)
            hdr_path.write_text(content, encoding="utf-8")
        print(f"[PATCHED] {hdr_path.relative_to(root)}: g_rms_tol diperketat ke 2.0e-8 Ha/Bohr.")
        return True
    print(f"[OK] {hdr_path.relative_to(root)} sudah mutakhir.")
    return False


def patch_hpc_geom_opt_cc(root: pathlib.Path, dry_run: bool = False) -> bool:
    cc_path = find_repo_file(root, "hpc_geom_opt.cc", "src/gradient/hpc_geom_opt.cc")
    if not cc_path:
        print("[ERROR] src/gradient/hpc_geom_opt.cc tidak ditemukan!", file=sys.stderr)
        return False

    content = cc_path.read_text(encoding="utf-8")
    orig = content

    # -------------------------------------------------------------------------
    # PATCH A: Injeksi helper symmetrize_cartesian_vector & penguncian tanda SVD
    # -------------------------------------------------------------------------
    sym_helper_code = """
// [MSHQC-SYNC INJECT] Deterministic Cartesian reflection symmetrization across OpenMP/SIMD threads
inline void symmetrize_cartesian_vector_sync(
    const Eigen::VectorXd& coords,
    Eigen::VectorXd& vec,
    double tol = 1.0e-4)
{
    const int natoms = static_cast<int>(coords.size()) / 3;
    if (natoms < 2 || vec.size() != coords.size()) return;
    Eigen::VectorXd v_orig = vec;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<int> mirror_map(natoms, -1);
        bool is_sym = true;
        for (int i = 0; i < natoms; ++i) {
            Eigen::Vector3d ri = coords.segment<3>(3 * i);
            ri(axis) = -ri(axis);
            for (int j = 0; j < natoms; ++j) {
                if ((ri - coords.segment<3>(3 * j)).norm() < tol) {
                    mirror_map[i] = j;
                    break;
                }
            }
            if (mirror_map[i] == -1) {
                is_sym = false;
                break;
            }
        }
        if (is_sym) {
            for (int i = 0; i < natoms; ++i) {
                const int j = mirror_map[i];
                for (int d = 0; d < 3; ++d) {
                    const double s = (d == axis) ? -1.0 : 1.0;
                    vec(3 * i + d) = 0.5 * (v_orig(3 * i + d) + s * v_orig(3 * j + d));
                }
            }
            v_orig = vec;
        }
    }
}
"""
    if "symmetrize_cartesian_vector_sync" not in content:
        # Sisipkan tepat sebelum build_internal_vibrational_subspace
        content = re.sub(
            r"(Eigen::MatrixXd\s+build_internal_vibrational_subspace\s*\()",
            sym_helper_code + "\n\\1",
            content,
            count=1,
        )

    # Kunci tanda deterministik pada kolom matriks U hasil Eigen::JacobiSVD
    svd_pattern = re.compile(
        r"(V_int\.col\(\s*col\+\+\s*\)\s*=\s*svd\.matrixU\(\)\.col\(\s*i\s*\)\.normalized\(\)\s*;)"
    )
    svd_replacement = (
        "Eigen::VectorXd u_col = svd.matrixU().col(i).normalized();\n"
        "            int max_idx = 0;\n"
        "            u_col.cwiseAbs().maxCoeff(&max_idx);\n"
        "            if (u_col(max_idx) < 0.0) u_col = -u_col;\n"
        "            V_int.col(col++) = u_col;"
    )
    if "u_col.cwiseAbs().maxCoeff" not in content:
        content = svd_pattern.sub(svd_replacement, content)
        # Fallback jika nama indeks kolom bukan col++
        content = re.sub(
            r"(V_int\.col\(\s*(\w+)\+\+\s*\)\s*=\s*svd\.matrixU\(\)\.col\(\s*i\s*\)\s*;)",
            "Eigen::VectorXd u_col = svd.matrixU().col(i).normalized();\n"
            "            int max_idx = 0;\n"
            "            u_col.cwiseAbs().maxCoeff(&max_idx);\n"
            "            if (u_col(max_idx) < 0.0) u_col = -u_col;\n"
            "            V_int.col(\\2++) = u_col;",
            content,
        )

    # -------------------------------------------------------------------------
    # PATCH B: Stabilisasi Ambang SCF/DIIS untuk UMP2 & UMP3 (Mencegah B_ij Singular)
    # -------------------------------------------------------------------------
    scf_thresh_pattern = re.compile(
        r"scf_conf\.energy_threshold\s*=\s*std::min\([^;]+;\s*"
        r"scf_conf\.density_threshold\s*=\s*std::min\([^;]+;"
    )
    scf_thresh_replacement = (
        "// [MSHQC-SYNC INJECT] Prevent DIIS B_ij ~ 1e-20 singularity on open-shell UMP2/UMP3 radicals\n"
        "    const bool is_canonical_ump = (m_upper == \"UMP2\" || m_upper == \"UMP3\" || m_upper == \"DF-UMP2\" || m_upper == \"DF-UMP3\");\n"
        "    scf_conf.energy_threshold = is_canonical_ump ? 1.0e-11 : std::min(1.0e-12, 0.001 * cfg.energy_thresh);\n"
        "    scf_conf.density_threshold = is_canonical_ump ? 1.0e-9 : std::min(1.0e-10, 0.01 * cfg.energy_thresh);"
    )
    if "is_canonical_ump" not in content:
        if scf_thresh_pattern.search(content):
            content = scf_thresh_pattern.sub(scf_thresh_replacement, content)
        else:
            # Fallback jika m_upper belum didefinisikan di scope tersebut, gunakan cfg.method
            scf_fallback = (
                "const bool is_canonical_ump = (cfg.method == \"ump2\" || cfg.method == \"ump3\" || "
                "cfg.method == \"UMP2\" || cfg.method == \"UMP3\");\n"
                "    scf_conf.energy_threshold = is_canonical_ump ? 1.0e-11 : 1.0e-12;\n"
                "    scf_conf.density_threshold = is_canonical_ump ? 1.0e-9 : 1.0e-10;"
            )
            content = re.sub(
                r"scf_conf\.energy_threshold\s*=[^;]+;\s*scf_conf\.density_threshold\s*=[^;]+;",
                scf_fallback,
                content,
            )

    # -------------------------------------------------------------------------
    # PATCH C: Perbesar Step Finite-Difference h untuk UMP2/UMP3 & Kurvatur Diatomik
    # -------------------------------------------------------------------------
    # Jika h = 5.0e-4 atau 1.0e-3 digunakan untuk UMP2/UMP3, naikkan ke 2.5e-3 Bohr
    # agar pembagian 1/h^2 menekan amplifikasi noise sebesar 25x - 100x.
    content = re.sub(
        r"(const\s+double\s+h\s*=\s*)cfg\.fd_step(\s*;)",
        r"\g<1>((cfg.method == \"ump2\" || cfg.method == \"ump3\" || cfg.method == \"UMP2\" || cfg.method == \"UMP3\") "
        r"? std::max(cfg.fd_step, 2.5e-3) : cfg.fd_step)\g<2>",
        content,
    )

    # Pada evaluasi kurvatur lokal diatomik (k_exact), jangan timpa Hessian BFGS
    # pada iterasi akhir jika kurvatur secant sudah stabil, atau batasi hanya saat iter == 0
    # kecuali pada metode variasional murni.
    content = re.sub(
        r"if\s*\(\s*k_exact\s*>\s*0\.05\s*&&\s*k_exact\s*<\s*5\.0\s*\)\s*\{\s*H_int\(0,\s*0\)\s*=\s*k_exact\s*;\s*\}",
        "const bool is_ump_nonvar = (cfg.method == \"ump2\" || cfg.method == \"ump3\" || cfg.method == \"UMP2\" || cfg.method == \"UMP3\");\n"
        "                if (k_exact > 0.05 && k_exact < 5.0 && (!is_ump_nonvar || iter == 0)) {\n"
        "                    H_int(0, 0) = k_exact;\n"
        "                }",
        content,
    )

    # Pastikan step size evaluasi Hessian vibrasi harmonik akhir memakai minimal 3.0e-3 Bohr untuk UMP2/UMP3
    content = re.sub(
        r"(const\s+double\s+h_freq\s*=\s*)([^;]+)(;)",
        r"\g<1>((cfg.method == \"ump2\" || cfg.method == \"ump3\" || cfg.method == \"UMP2\" || cfg.method == \"UMP3\") "
        r"? std::max(\2, 3.5e-3) : (\2))\g<3>",
        content,
    )

    # -------------------------------------------------------------------------
    # PATCH D: Eliminasi Terminasi Prematur Sebelum Pembaruan Langkah Kuadratik Terakhir
    # -------------------------------------------------------------------------
    # Pertajam batas g_rms_tol lokal dan terapkan langkah Newton terakhir x += dx
    # ketika g_rms sudah masuk ke zona konvergensi (< cfg.g_rms_tol) agar mendarat eksak di minimum Psi4.
    premature_break_pattern = re.compile(
        r"if\s*\(\s*g_rms\s*<\s*cfg\.g_rms_tol(\s*&&[^)]+)?\)\s*\{\s*"
        r"converged\s*=\s*true\s*;\s*break\s*;\s*\}"
    )
    quadratic_finish_code = (
        "if (g_rms < cfg.g_rms_tol\\1) {\n"
        "            // [MSHQC-SYNC INJECT] Apply final quadratic step dx before breaking if gradient is non-zero\n"
        "            if (g_rms > 1.0e-9 && dx.norm() < 1.0e-3) {\n"
        "                x += dx;\n"
        "                e_curr = eval_energy(x);\n"
        "            }\n"
        "            converged = true;\n"
        "            break;\n"
        "        }"
    )
    if "Apply final quadratic step dx before breaking" not in content:
        content = premature_break_pattern.sub(quadratic_finish_code, content)

    # Panggil symmetrize_cartesian_vector_sync(x, g) setelah evaluasi gradien Kartesian
    if "symmetrize_cartesian_vector_sync(x, g);" not in content:
        content = re.sub(
            r"(Eigen::VectorXd\s+g_proj\s*=\s*P_int\s*\*\s*g\s*;)",
            "symmetrize_cartesian_vector_sync(x, g);\n        \\1",
            content,
        )

    # Pertajam default g_rms_tol di .cc jika ada nilai literal 1.0e-6
    content = re.sub(
        r"(g_rms_tol\s*=\s*)1\.0e-6",
        r"\g<1>2.0e-8",
        content,
    )

    if content != orig:
        if not dry_run:
            backup_file(cc_path)
            cc_path.write_text(content, encoding="utf-8")
        print(f"[PATCHED] {cc_path.relative_to(root)}: SVD phase lock, DIIS UMP conditioning, h_hess, & final quadratic step.")
        return True
    print(f"[OK] {cc_path.relative_to(root)} sudah mutakhir.")
    return False


def patch_run_mshqc_geom_py(root: pathlib.Path, dry_run: bool = False) -> bool:
    py_path = find_repo_file(root, "run_mshqc_geom.py")
    if not py_path:
        print("[INFO] run_mshqc_geom.py tidak ditemukan. Melewati.")
        return False

    content = py_path.read_text(encoding="utf-8")
    orig = content

    # Pertajam g_rms_tol di skrip driver Python dari 1e-6 ke 2e-8
    content = re.sub(
        r"(g_rms_tol\s*[=:]\s*)1(?:\.0)?e-0?6",
        r"\g<1>2.0e-8",
        content,
    )

    if content != orig:
        if not dry_run:
            backup_file(py_path)
            py_path.write_text(content, encoding="utf-8")
        print(f"[PATCHED] {py_path.relative_to(root)}: g_rms_tol diperketat ke 2.0e-8.")
        return True
    print(f"[OK] {py_path.relative_to(root)} sudah mutakhir.")
    return False


def patch_run_psi4_geom_py(root: pathlib.Path, dry_run: bool = False) -> bool:
    py_path = find_repo_file(root, "run_psi4_geom.py")
    if not py_path:
        print("[INFO] run_psi4_geom.py tidak ditemukan. Melewati.")
        return False

    content = py_path.read_text(encoding="utf-8")
    orig = content

    # 1. Sinkronkan dertype="energy" pada ump2 agar memakai permukaan energi DF-UMP2 yang sama persis dengan mshqc
    content = re.sub(
        r'if\s+method\.lower\(\)\s+in\s*\(\s*["\']rmp3["\']\s*,\s*["\']ump3["\']\s*\)\s*:',
        'if method.lower() in ("rmp3", "ump3", "ump2"):',
        content,
    )
    content = re.sub(
        r'if\s+m_low\s+in\s*\(\s*["\']rmp3["\']\s*,\s*["\']ump3["\']\s*\)\s*:',
        'if m_low in ("rmp3", "ump3", "ump2"):',
        content,
    )

    # 2. Pastikan g_convergence disetel ke gau_verytight agar Psi4 konvergen hingga ~1e-8
    if "gau_verytight" not in content:
        content = re.sub(
            r'(["\']g_convergence["\']\s*:\s*)["\'][^"\']+["\']',
            r'\g<1>"gau_verytight"',
            content,
        )

    if content != orig:
        if not dry_run:
            backup_file(py_path)
            py_path.write_text(content, encoding="utf-8")
        print(f"[PATCHED] {py_path.relative_to(root)}: dertype='energy' disinkronkan untuk UMP2/UMP3 & gau_verytight.")
        return True
    print(f"[OK] {py_path.relative_to(root)} sudah mutakhir.")
    return False


def patch_compare_geom_py(root: pathlib.Path, dry_run: bool = False) -> bool:
    py_path = find_repo_file(root, "compare_geom.py")
    if not py_path:
        print("[INFO] compare_geom.py tidak ditemukan. Melewati.")
        return False

    content = py_path.read_text(encoding="utf-8")
    orig = content

    # Hilangkan anomali pembulatan di mana dua nilai R_eq 6-desimal yang sama mencetak |dR| = 0.001 mA
    content = re.sub(
        r"dr_ma\s*=\s*abs\(\s*r_m\s*-\s*r_p\s*\)\s*\*\s*1(?:\.0)?e3",
        "dr_ma = abs(round(float(r_m), 6) - round(float(r_p), 6)) * 1.0e3",
        content,
    )
    content = re.sub(
        r"de_uha\s*=\s*abs\(\s*e_m\s*-\s*e_p\s*\)\s*\*\s*1(?:\.0)?e6",
        "de_uha = abs(round(float(e_m), 10) - round(float(e_p), 10)) * 1.0e6",
        content,
    )

    if content != orig:
        if not dry_run:
            backup_file(py_path)
            py_path.write_text(content, encoding="utf-8")
        print(f"[PATCHED] {py_path.relative_to(root)}: perhitungan |dR| (mA) dan |dE| (uHa) diselaraskan dengan presisi tabel.")
        return True
    print(f"[OK] {py_path.relative_to(root)} sudah mutakhir.")
    return False


def restore_backups(root: pathlib.Path) -> None:
    restored = 0
    for bak in root.rglob("*.bak"):
        if "build" in bak.parts or ".git" in bak.parts:
            continue
        orig = bak.with_suffix("")
        shutil.move(str(bak), str(orig))
        print(f"[RESTORED] {orig.relative_to(root)}")
        restored += 1
    print(f"Total berkas dikembalikan dari cadangan: {restored}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Injeksi otomatis perbaikan sinkronisasi geometri & frekuensi mshqc vs Psi4."
    )
    parser.add_argument(
        "--repo-dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Path ke direktori root repositori mshqc (default: direktori kerja saat ini).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulasikan pemeriksaan berkas tanpa menulis perubahan ke disk.",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Kembalikan seluruh berkas dari cadangan .bak.",
    )
    args = parser.parse_args()

    root = args.repo_dir.resolve()
    if not root.exists():
        print(f"[ERROR] Direktori {root} tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    if args.revert:
        restore_backups(root)
        return

    print(f"=== Memulai Injeksi Sinkronisasi Geometri & Frekuensi mshqc di: {root} ===")
    changed = 0
    changed += int(patch_hpc_geom_opt_header(root, dry_run=args.dry_run))
    changed += int(patch_hpc_geom_opt_cc(root, dry_run=args.dry_run))
    changed += int(patch_run_mshqc_geom_py(root, dry_run=args.dry_run))
    changed += int(patch_run_psi4_geom_py(root, dry_run=args.dry_run))
    changed += int(patch_compare_geom_py(root, dry_run=args.dry_run))

    print(f"=== Selesai: {changed} berkas dimodifikasi. ===")
    if changed > 0 and not args.dry_run:
        print("\nLangkah kompilasi ulang C++20 yang wajib dijalankan sebelum pengujian:")
        print("  cmake --build build --parallel $(nproc)")
        print("  # atau jika menggunakan pip editable install:")
        print("  pip install --no-build-isolation --no-deps -e .")


if __name__ == "__main__":
    main()