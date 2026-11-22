#!/usr/bin/env python3
"""Menyelesaikan 3 patch terakhir Fase 4 (calculators.py, methods/__init__.py, workflow CI)."""

from pathlib import Path
import re
import shutil
import sys

ROOT = Path(".").resolve()


def backup_file(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak_phase4_fix")
        if not bak.exists():
            shutil.copy2(path, bak)


def patch_calculators() -> None:
    calc = ROOT / "python/mshqc/calculators.py"
    if not calc.exists():
        sys.exit("[ERROR] python/mshqc/calculators.py tidak ditemukan.")
    c_text = calc.read_text(encoding="utf-8")

    if "INJECTED_HYBRID_PHASE4_CACHE" in c_text:
        print("[SKIP ] python/mshqc/calculators.py sudah ter-patch.")
        return

    if "_ao_factor_cache" not in c_text:
        c_text, n_init = re.subn(
            r"([ \t]*self\.scf_res = None\n[ \t]*self\.mp2_res = None)",
            r"\1\n        self._ao_factor_cache = None",
            c_text,
            count=1,
        )
        if n_init != 1:
            sys.exit("[ERROR] Baris self.scf_res / self.mp2_res tidak ditemukan di calculators.py.")

    pattern = re.compile(
        r"(?:^[ \t]*# \[INJECTED_HYBRID_PHASE3_CALCULATORS\]\n)?"
        r"^[ \t]*def _run_hybrid_omp_method\(self, method_name: str, scf_res: Optional\[mshqc\.SCFResult\] = None, grad_thresh: float = 1e-5, max_iter: int = 50\):\n"
        r"[ \t]*from \.methods\.interop import run_from_arrays\n"
        r"[ \t]*if self\.eri_method != \"df\":\n"
        r"[ \t]*raise ValueError\([^\n]+\)\n"
        r"[ \t]*if scf_res is None:\n"
        r"[ \t]*scf_res = self\.scf_res or self\.run_scf\(\)\n"
        r"[ \t]*aux_basis = mshqc\.BasisSet\(self\.aux_basis_name, self\.molecule, self\.basis_dir\)\n"
        r"[ \t]*B_ao = mshqc\.kernels\.compute_df_ao_factor\(self\.molecule, self\.basis, aux_basis, self\.threshold, False\)",
        re.MULTILINE,
    )

    replacement = (
        '    # [INJECTED_HYBRID_PHASE4_CACHE]\n'
        '    def _get_ao_3index_factor(self, scf_res: mshqc.SCFResult):\n'
        '        if hasattr(scf_res, "L_mat") and scf_res.L_mat.shape[0] > 0 and scf_res.L_mat.shape[1] > 0:\n'
        '            return scf_res.L_mat\n'
        '        if self._ao_factor_cache is not None:\n'
        '            return self._ao_factor_cache\n'
        '        if self.eri_method == "df":\n'
        '            aux_basis = mshqc.BasisSet(self.aux_basis_name, self.molecule, self.basis_dir)\n'
        '            self._ao_factor_cache = mshqc.kernels.compute_df_ao_factor(\n'
        '                self.molecule, self.basis, aux_basis, self.threshold, False, self.print_level > 0\n'
        '            )\n'
        '            return self._ao_factor_cache\n'
        '        if self.eri_method == "cholesky":\n'
        '            chol = mshqc.CholeskyERI(self.basis, self.integrals)\n'
        '            chol.set_threshold(self.threshold)\n'
        '            chol.set_print_level(self.print_level)\n'
        '            chol.compute()\n'
        '            self._ao_factor_cache = chol.get_L_mat().copy()\n'
        '            return self._ao_factor_cache\n'
        '        raise ValueError(f"Metode hibrida mendukung eri_method=\'df\' atau \'cholesky\', diterima: {self.eri_method}")\n\n'
        '    def _run_hybrid_omp_method(self, method_name: str, scf_res: Optional[mshqc.SCFResult] = None, grad_thresh: float = 1e-5, max_iter: int = 50):\n'
        '        from .methods.interop import run_from_arrays\n'
        '        if self.eri_method not in ("df", "cholesky"):\n'
        '            raise ValueError(f"Metode hibrida \'{method_name}\' mendukung eri_method=\'df\' atau \'cholesky\'.")\n'
        '        if scf_res is None:\n'
        '            scf_res = self.scf_res or self.run_scf()\n'
        '        B_ao = self._get_ao_3index_factor(scf_res)'
    )

    updated, n_rep = pattern.subn(replacement, c_text, count=1)
    if n_rep != 1:
        sys.exit("[ERROR] Blok _run_hybrid_omp_method masih tidak cocok di calculators.py.")

    backup_file(calc)
    calc.write_text(updated, encoding="utf-8")
    print("[PATCH] python/mshqc/calculators.py (reuse L_mat + dukungan Cholesky)")


def patch_methods_init() -> None:
    m_init = ROOT / "python/mshqc/methods/__init__.py"
    if not m_init.exists():
        sys.exit("[ERROR] python/mshqc/methods/__init__.py tidak ditemukan.")
    mi_text = m_init.read_text(encoding="utf-8")
    if "optimize_orbitals_rhf" in mi_text:
        print("[SKIP ] python/mshqc/methods/__init__.py sudah mengekspor optimize_orbitals_rhf.")
        return

    mi_text = mi_text.replace(
        "from .interop import run_from_arrays, run_from_pyscf",
        "from .interop import run_from_arrays, run_from_pyscf\nfrom .custom import optimize_orbitals_rhf",
        1,
    ).replace(
        '    "run_from_arrays", "run_from_pyscf",',
        '    "run_from_arrays", "run_from_pyscf",\n    "optimize_orbitals_rhf",',
        1,
    )
    backup_file(m_init)
    m_init.write_text(mi_text, encoding="utf-8")
    print("[PATCH] python/mshqc/methods/__init__.py")


def patch_workflow() -> None:
    wf = ROOT / ".github/workflows/test_mshqc_hybrid.yml"
    if not wf.exists():
        print("[SKIP ] .github/workflows/test_mshqc_hybrid.yml tidak ditemukan.")
        return
    text = wf.read_text(encoding="utf-8")
    if "test_hybrid_phase4.py" in text:
        print("[SKIP ] .github/workflows/test_mshqc_hybrid.yml sudah menyertakan Fase 4.")
        return

    step3_pat = re.compile(
        r'(^[ \t]*- name: "\[03/03\] Run Hybrid Phase 3 Verification \(test_hybrid_phase3\.py\)"\n'
        r'[ \t]*shell: bash -el \{0\}\n'
        r'[ \t]*run: \|\n'
        r'[ \t]*set -o pipefail\n'
        r'[ \t]*micromamba activate geom_env\n'
        r'[ \t]*python -m pytest tests/test_hybrid_phase3\.py -v -s 2>&1 \| tee 03_test_hybrid_phase3_log\.txt)',
        re.MULTILINE,
    )
    m = step3_pat.search(text)
    if not m:
        print("[WARN ] Anchor step Phase 3 di workflow tidak persis; lewati patch CI.")
        return

    step4_block = (
        m.group(1).replace("[03/03]", "[03/04]")
        + '\n\n'
        + '      - name: "[04/04] Run Hybrid Phase 4 Verification (test_hybrid_phase4.py)"\n'
        + '        shell: bash -el {0}\n'
        + '        run: |\n'
        + '          set -o pipefail\n'
        + '          micromamba activate geom_env\n'
        + '          python -m pytest tests/test_hybrid_phase4.py -v -s 2>&1 | tee 04_test_hybrid_phase4_log.txt'
    )
    updated = (
        text[:m.start()] + step4_block + text[m.end():]
    ).replace("[01/03]", "[01/04]", 1).replace("[02/03]", "[02/04]", 1)

    updated += (
        "\n      - name: Upload Hybrid Phase 4 Test Log\n"
        "        if: always()\n"
        "        uses: actions/upload-artifact@v4\n"
        "        with:\n"
        "          name: mshqc-hybrid-phase4-log-${{ matrix.os }}\n"
        "          path: 04_test_hybrid_phase4_log.txt\n"
    )
    backup_file(wf)
    wf.write_text(updated, encoding="utf-8")
    print("[PATCH] .github/workflows/test_mshqc_hybrid.yml")


def main() -> None:
    patch_calculators()
    patch_methods_init()
    patch_workflow()
    print("\n[DONE] Sisa patch Fase 4 berhasil diterapkan.")


if __name__ == "__main__":
    main()