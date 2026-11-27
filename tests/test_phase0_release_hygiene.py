"""
Phase 0 CPC release hygiene gate (aligned with .modal/gpu_benchmark.py).
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
FORBIDDEN_SYMBOLS = ("oo_dh_dft", "run_oo_double_hybrid_rhf", "OODoubleHybridResult")


def _iter_release_py_files() -> list[Path]:
    files: list[Path] = []
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in IGNORED for part in p.relative_to(REPO_ROOT).parts):
            continue
        if p.name.startswith("inject_") or p.name == "inject.py" or p == Path(__file__).resolve():
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
    assert not leaked_files, f"Berkas eksperimental masih di tree rilis: {leaked_files}"


def test_no_public_import_or_symbol_of_oo_dh_dft() -> None:
    offenders: list[str] = []
    for p in _iter_release_py_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(sym in stripped for sym in FORBIDDEN_SYMBOLS):
                offenders.append(f"{p.relative_to(REPO_ROOT)}:{lineno}: {stripped}")
    assert not offenders, "Referensi simbol oo_dh_dft masih aktif di kode/tes rilis:\n" + "\n".join(offenders)


def test_single_canonical_modal_gpu_runner_exists() -> None:
    canonical = REPO_ROOT / ".modal" / "gpu_benchmark.py"
    duplicate = REPO_ROOT / "scripts" / "modal_gpu_bench.py"
    assert canonical.exists(), ".modal/gpu_benchmark.py tidak ditemukan."
    assert not duplicate.exists(), "Hapus runner duplikat scripts/modal_gpu_bench.py."
    text = canonical.read_text(encoding="utf-8")
    assert "torch.float64" in text
    assert '"-e", "."' in text
    assert "cuda_gpu_benchmark_raw.json" in text
