"""
Phase 0 CPC release hygiene gate.

Memastikan:
1. Manifest ruang lingkup rilis CPC (`repro_cpc/CPC_RELEASE_SCOPE.md`) ada.
2. Modul eksperimental `oo_dh_dft` tidak berada di tree sumber rilis CPC.
3. Tidak ada impor `oo_dh_dft` pada kode paket atau suite tes rilis.
4. Runner Modal GPU membangun `mshqc` dari source lokal (bukan PyPI) dan
   memakai backend PyTorch CUDA FP64.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED = {
    ".git",
    "build",
    "build-asan",
    "build-tsan",
    "dist",
    ".venv",
    "venv",
    "__pycache__",
    "experimental",
    "artifacts",
}


def _iter_release_py_files() -> list[Path]:
    files: list[Path] = []
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in IGNORED for part in p.relative_to(REPO_ROOT).parts):
            continue
        if p.name.startswith("inject_phase0_") or p == Path(__file__).resolve():
            continue
        files.append(p)
    return files


def test_cpc_release_scope_manifest_exists() -> None:
    manifest = REPO_ROOT / "repro_cpc" / "CPC_RELEASE_SCOPE.md"
    assert manifest.exists(), "Manifest repro_cpc/CPC_RELEASE_SCOPE.md tidak ditemukan."
    assert "Feature Freeze" in manifest.read_text(encoding="utf-8")


def test_no_experimental_oo_dh_dft_in_release_tree() -> None:
    leaked_files = [
        str(p.relative_to(REPO_ROOT))
        for p in _iter_release_py_files()
        if "oo_dh_dft" in p.name
    ]
    assert not leaked_files, (
        f"Berkas eksperimental oo_dh_dft masih ada di tree rilis: {leaked_files}. "
        "Pindahkan ke experimental/oo_dh_dft/."
    )


def test_no_public_import_of_oo_dh_dft() -> None:
    offenders: list[str] = []
    for p in _iter_release_py_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "oo_dh_dft" in stripped and ("import " in stripped or "from " in stripped):
                offenders.append(f"{p.relative_to(REPO_ROOT)}:{lineno}: {stripped}")
    assert not offenders, "Ditemukan impor oo_dh_dft pada kode rilis:\n" + "\n".join(offenders)


def test_modal_runner_uses_source_build_and_torch_fp64() -> None:
    runner = REPO_ROOT / "scripts" / "modal_gpu_bench.py"
    assert runner.exists(), "scripts/modal_gpu_bench.py tidak ditemukan."
    text = runner.read_text(encoding="utf-8")
    assert "torch.float64" in text, "Modal runner harus memverifikasi torch.float64."
    assert '"-e", "."' in text, "Modal runner harus membangun mshqc dari source tree lokal."
    assert "pip install mshqc" not in text, "Modal runner tidak boleh menginstal mshqc dari PyPI."
