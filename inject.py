#!/usr/bin/env python3
"""
inject_fix_mshqc_sync.py
Perbaikan otomatis:
1. python/mshqc/calculators.py (fix TypeError NoneType pg/pl saat use_symmetry=False)
2. .github/workflows/heavy_large_benchmark.yml (fix micromamba: command not found)
3. .github/workflows/test_geom_only.yml (nonaktifkan auto-trigger push, sisakan workflow_dispatch)
"""

import re
import sys
from pathlib import Path

ROOT = Path.cwd()


def patch_calculators():
    path = ROOT / "python" / "mshqc" / "calculators.py"
    if not path.is_file():
        print(f"[SKIP] Tidak ditemukan: {path}")
        return
    txt = path.read_text(encoding="utf-8")

    # 1. Patch run_scf
    old_scf = """    def run_scf(self, ref: Optional[str] = None) -> mshqc.SCFResult:
        ref_mode = (ref or self.reference).lower()
        conf = self._make_scf_config()
        if ref_mode == "rhf":
            solver = mshqc.RHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, conf)
        elif ref_mode == "rohf":
            solver = mshqc.ROHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
        else:
            solver = mshqc.UHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
        self.scf_res = solver.compute()
        return self.scf_res"""

    new_scf = """    def run_scf(self, ref: Optional[str] = None) -> mshqc.SCFResult:
        ref_mode = (ref or self.reference).lower()
        conf = self._make_scf_config()
        use_sym = (self.pg is not None and self.pl is not None)
        if ref_mode == "rhf":
            if use_sym:
                solver = mshqc.RHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, conf)
            else:
                solver = mshqc.RHF(self.molecule, self.basis, self.integrals, conf)
        elif ref_mode == "rohf":
            if use_sym:
                solver = mshqc.ROHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
            else:
                solver = mshqc.ROHF(self.molecule, self.basis, self.integrals, self.n_alpha, self.n_beta, conf)
        else:
            if use_sym:
                solver = mshqc.UHF(self.molecule, self.basis, self.integrals, self.pg, self.pl, self.n_alpha, self.n_beta, conf)
            else:
                solver = mshqc.UHF(self.molecule, self.basis, self.integrals, self.n_alpha, self.n_beta, conf)
        self.scf_res = solver.compute()
        return self.scf_res"""

    # 2. Patch run_mp2, run_omp2, run_omp3 agar aman saat self.pg/self.pl == None
    old_mp2 = """        if self.reference == "rhf":
            solver = mshqc.RMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)
        elif self.reference == "rohf":
            solver = mshqc.ROMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)
        else:
            solver = mshqc.UMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)"""

    new_mp2 = """        sym_args = (self.pg, self.pl) if (self.pg is not None and self.pl is not None) else ()
        if self.reference == "rhf":
            solver = mshqc.RMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
        elif self.reference == "rohf":
            solver = mshqc.ROMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)
        else:
            solver = mshqc.UMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)"""

    old_omp2 = "solver = mshqc.OMP2(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)"
    new_omp2 = (
        "sym_args = (self.pg, self.pl) if (self.pg is not None and self.pl is not None) else ()\n"
        "        solver = mshqc.OMP2(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)"
    )

    old_omp3 = "solver = mshqc.OMP3(self.molecule, self.basis, self.integrals, scf_res, conf, self.pg, self.pl)"
    new_omp3 = (
        "sym_args = (self.pg, self.pl) if (self.pg is not None and self.pl is not None) else ()\n"
        "        solver = mshqc.OMP3(self.molecule, self.basis, self.integrals, scf_res, conf, *sym_args)"
    )

    changed = False
    for old, new, label in (
        (old_scf, new_scf, "run_scf"),
        (old_mp2, new_mp2, "run_mp2"),
        (old_omp2, new_omp2, "run_omp2"),
        (old_omp3, new_omp3, "run_omp3"),
    ):
        if old in txt:
            txt = txt.replace(old, new, 1)
            changed = True
            print(f"[PATCH] calculators.py -> {label} berhasil diperbaiki.")

    if changed:
        path.write_text(txt, encoding="utf-8")
    else:
        print("[INFO] calculators.py sudah ter-patch sebelumnya.")


def patch_heavy_large_workflow():
    path = ROOT / ".github" / "workflows" / "heavy_large_benchmark.yml"
    path.parent.mkdir(parents=True, exist_ok=True)

    yml = """name: Heavy Atoms & Large Molecules Benchmark (mshqc vs Psi4)

on:
  push:
    branches:
      - build-pypi
      - test-geom
      - test
      - main
  workflow_dispatch:

env:
  OMP_NUM_THREADS: "4"
  OMP_WAIT_POLICY: "PASSIVE"
  TBLIS_NUM_THREADS: "4"
  MKL_NUM_THREADS: "1"
  OPENBLAS_NUM_THREADS: "1"
  HDF5_USE_FILE_LOCKING: "FALSE"
  PYTHONUNBUFFERED: "1"
  PYTHONIOENCODING: "utf-8"

jobs:
  # ===========================================================================
  # JALUR PARALEL 1: BUILD mshqc & RUN run_mshqc_heavy_large.py
  # ===========================================================================
  mshqc_heavy_large:
    name: "[Parallel 1/2] Build & Run mshqc Heavy/Large (${{ matrix.os }})"
    runs-on: ${{ matrix.os }}
    timeout-minutes: 240
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - name: Checkout Source Code
        uses: actions/checkout@v4

      - name: Fetch Basis Sets from mshqc-basis
        shell: bash -eo pipefail {0}
        run: |
          git clone --depth 1 https://github.com/syahrulhidayat/mshqc-basis.git /tmp/mshqc-basis
          mkdir -p data/basis python/mshqc/basis
          cp -rf /tmp/mshqc-basis/data/basis/* data/basis/
          cp -rf /tmp/mshqc-basis/data/basis/* python/mshqc/basis/
          rm -rf /tmp/mshqc-basis

      - name: Set up Micromamba Build Environment for mshqc (geom_env)
        uses: mamba-org/setup-micromamba@v2
        with:
          init-shell: bash
          environment-name: geom_env
          create-args: >-
            -c conda-forge
            python=3.11
            cmake
            ninja
            make
            pkg-config
            cxx-compiler
            c-compiler
            llvm-openmp
            eigen
            hdf5=1.14.3
            openblas
            liblapacke
            nanobind
            setuptools
            wheel
            pip
            numpy
            scipy
            pytest

      - name: Restore Cached libcint & TBLIS (${{ runner.os }})
        id: cache-deps
        uses: actions/cache@v4
        with:
          path: .cache_local_deps
          key: local-geom-deps-${{ runner.os }}-v2

      - name: Build libcint & TBLIS if Not Cached
        shell: bash -el {0}
        run: |
          set -o pipefail
          eval "$("$MAMBA_EXE" shell hook --shell bash)"
          micromamba activate geom_env
          mkdir -p .cache_local_deps/include .cache_local_deps/lib
          if [ -f ".cache_local_deps/lib/libtblis.a" ] && ( [ -f ".cache_local_deps/lib/libcint.a" ] || [ -f ".cache_local_deps/lib64/libcint.a" ] ); then
            cp -rf .cache_local_deps/include/* "$CONDA_PREFIX/include/"
            cp -rf .cache_local_deps/lib/* "$CONDA_PREFIX/lib/" 2>/dev/null || true
            if [ -d ".cache_local_deps/lib64" ]; then
              cp -rf .cache_local_deps/lib64/* "$CONDA_PREFIX/lib/" 2>/dev/null || true
            fi
          else
            git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint
            cmake -S /tmp/libcint -B /tmp/libcint/build -GNinja \\
              -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \\
              -DCMAKE_POSITION_INDEPENDENT_CODE=ON \\
              -DCMAKE_C_FLAGS="-O3 -fPIC -std=gnu17 -Wno-incompatible-pointer-types -Wno-implicit-function-declaration" \\
              -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=OFF
            cmake --build /tmp/libcint/build -j4
            cmake --install /tmp/libcint/build
            if [ -f "$CONDA_PREFIX/lib64/libcint.a" ] && [ ! -f "$CONDA_PREFIX/lib/libcint.a" ]; then
              cp -f "$CONDA_PREFIX/lib64/libcint.a" "$CONDA_PREFIX/lib/libcint.a"
            fi

            git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis
            cd /tmp/tblis
            git submodule update --init --recursive

            python - << 'EOF'
          import pathlib
          p = pathlib.Path('src/external/catch/catch.hpp')
          p.write_text(p.read_text().replace('SIGSTKSZ', '8192'))
          for f in ('src/external/marray/include/short_vector.hpp', 'src/external/marray/include/range.hpp', 'src/util/thread.h'):
              fp = pathlib.Path(f)
              fp.write_text('#include <limits>\\n' + fp.read_text())
          tp = pathlib.Path('src/external/tci/tci/task_set.c')
          tp.write_text('#include <stdlib.h>\\n' + tp.read_text())
          ep = pathlib.Path('src/external/marray/include/expression.hpp')
          ep.write_text(ep.read_text().replace(
              '#include <x86intrin.h>',
              '#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))\\n#include <x86intrin.h>\\n#else\\n#undef __AVX512F__\\n#undef __AVX__\\n#undef __SSE2__\\n#endif'
          ))
          cp = pathlib.Path('src/util/cpuid.hpp')
          if cp.exists():
              txt = cp.read_text()
              txt = txt.replace('enum {VENDOR_ARM, VENDOR_UNKNOWN}', 'enum {VENDOR_ARM, VENDOR_UNKNOWN};')
              txt = txt.replace('enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN}', 'enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN};')
              cp.write_text(txt)
          cx = pathlib.Path('src/util/cpuid.cxx')
          if cx.exists():
              cx.write_text(cx.read_text().replace('strtoi(ptno, pos, 16)', '((void)pos, (int)strtol(ptno.c_str(), NULL, 16))'))
          EOF

            CFG_ARCH="reference"
            if [ "$(uname -m)" = "x86_64" ]; then
              CFG_ARCH="haswell,zen,core2,reference"
            fi
            ./configure --prefix="$CONDA_PREFIX" --enable-static --disable-shared --with-pic --without-hwloc \\
              --enable-config="$CFG_ARCH" \\
              CFLAGS="-O3 -fPIC -std=gnu17 -Wno-implicit-function-declaration -Wno-incompatible-function-pointer-types" \\
              CXXFLAGS="-O3 -fPIC -std=c++17 -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE"
            make -j4 bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS=
            make install bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS=

            cd "$GITHUB_WORKSPACE"
            cp -rf "$CONDA_PREFIX/include/cint"*.h "$CONDA_PREFIX/include/tci"* "$CONDA_PREFIX/include/tblis" .cache_local_deps/include/
            cp -rf "$CONDA_PREFIX/lib/libcint.a" "$CONDA_PREFIX/lib/libtblis.a" "$CONDA_PREFIX/lib/libtci.a" .cache_local_deps/lib/
          fi

      - name: Compile mshqc In-Place in geom_env
        shell: bash -el {0}
        run: |
          set -o pipefail
          eval "$("$MAMBA_EXE" shell hook --shell bash)"
          micromamba activate geom_env
          export TBLIS_DIR="$CONDA_PREFIX"
          export OpenMP_ROOT="$CONDA_PREFIX"
          export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
          export CMAKE_ARGS="-DOpenMP_ROOT=$CONDA_PREFIX"
          export CMAKE_BUILD_PARALLEL_LEVEL=4
          python -m pip install --no-build-isolation --no-deps -e . -v
          python -c "import mshqc; print('Compiled mshqc version:', getattr(mshqc, '__version__', 'local-build'))"

      - name: Run mshqc Heavy & Large Molecule Suite (run_mshqc_heavy_large.py)
        shell: bash -el {0}
        run: |
          set -o pipefail
          eval "$("$MAMBA_EXE" shell hook --shell bash)"
          micromamba activate geom_env
          python -u tests/run_mshqc_heavy_large.py 2>&1 | tee mshqc_heavy_large_running_log.txt
          test -s mshqc_heavy_large_raw.json

      - name: Upload Partial mshqc Outputs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: part-mshqc-heavy-large-${{ matrix.os }}
          retention-days: 1
          if-no-files-found: ignore
          path: |
            mshqc_heavy_large_running_log.txt
            mshqc_heavy_large_raw.json

  # ===========================================================================
  # JALUR PARALEL 2: SETUP psi4_env & RUN run_psi4_heavy_large.py
  # ===========================================================================
  psi4_heavy_large:
    name: "[Parallel 2/2] Run Psi4 Heavy/Large (${{ matrix.os }})"
    runs-on: ${{ matrix.os }}
    timeout-minutes: 240
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - name: Checkout Source Code
        uses: actions/checkout@v4

      - name: Set up Clean Psi4 Environment (psi4_env)
        uses: mamba-org/setup-micromamba@v2
        with:
          init-shell: bash
          environment-name: psi4_env
          create-args: >-
            -c conda-forge
            python=3.11
            psi4
            numpy
            scipy
            pip

      - name: Patch LibXC Table & Verify Psi4
        shell: bash -el {0}
        run: |
          set -o pipefail
          eval "$("$MAMBA_EXE" shell hook --shell bash)"
          micromamba activate psi4_env
          if ! python -c "import psi4" >/dev/null 2>&1; then
            echo "Mendeteksi ketidakcocokan fungsional LibXC pada Psi4, menerapkan patch proc_table.py..."
            python - << 'EOF'
          import pathlib, re, sys
          for p in pathlib.Path(sys.prefix).rglob("psi4/driver/procrouting/proc_table.py"):
              txt = p.read_text(encoding="utf-8")
              pattern = r"for key in functionals:\\s+ssuper = build_superfunctional_from_dictionary\\(functionals\\[key\\], 1, 1, True\\)\\[0\\]"
              replacement = (
                  "for key in list(functionals.keys()):\\n"
                  "    try:\\n"
                  "        ssuper = build_superfunctional_from_dictionary(functionals[key], 1, 1, True)[0]\\n"
                  "    except Exception:\\n"
                  "        functionals.pop(key, None)\\n"
                  "        continue"
              )
              new_txt, count = re.subn(pattern, replacement, txt)
              if count > 0:
                  p.write_text(new_txt, encoding="utf-8")
                  print(f"Berhasil melakukan patch proc_table.py pada: {p}")
          EOF
          fi
          python -c "import psi4; print('Psi4 berhasil dimuat di psi4_env, versi:', psi4.__version__)"

      - name: Run Psi4 Heavy & Large Molecule Reference Suite (run_psi4_heavy_large.py)
        shell: bash -el {0}
        run: |
          set -o pipefail
          eval "$("$MAMBA_EXE" shell hook --shell bash)"
          micromamba activate psi4_env
          python -u tests/run_psi4_heavy_large.py 2>&1 | tee psi4_heavy_large_running_log.txt
          test -s psi4_heavy_large_raw.json

      - name: Upload Partial Psi4 Outputs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: part-psi4-heavy-large-${{ matrix.os }}
          retention-days: 1
          if-no-files-found: ignore
          path: |
            psi4_heavy_large_running_log.txt
            psi4_heavy_large_raw.json

  # ===========================================================================
  # TAHAP AKHIR: COMPARE HEAD-TO-HEAD & GABUNGKAN SEMUA ARTIFACT
  # ===========================================================================
  compare_heavy_large:
    name: "[Compare] Heavy & Large mshqc vs Psi4 (${{ matrix.os }})"
    needs: [mshqc_heavy_large, psi4_heavy_large]
    if: always()
    runs-on: ${{ matrix.os }}
    timeout-minutes: 15
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - name: Checkout Source Code
        uses: actions/checkout@v4

      - name: Download mshqc Heavy & Large Outputs
        uses: actions/download-artifact@v4
        continue-on-error: true
        with:
          name: part-mshqc-heavy-large-${{ matrix.os }}

      - name: Download Psi4 Heavy & Large Outputs
        uses: actions/download-artifact@v4
        continue-on-error: true
        with:
          name: part-psi4-heavy-large-${{ matrix.os }}

      - name: Compare Heavy & Large Molecule Tables (compare_heavy_large.py)
        shell: bash -el {0}
        run: |
          set -o pipefail
          test -s mshqc_heavy_large_raw.json
          test -s psi4_heavy_large_raw.json
          python3 -u tests/compare_heavy_large.py mshqc_heavy_large_raw.json psi4_heavy_large_raw.json 2>&1 | tee compare_heavy_large_log.txt
          grep -q "Total Matched Heavy/Large Runs: 8" compare_heavy_large_log.txt

      - name: Upload Combined Heavy & Large Logs & JSONs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: mshqc-vs-psi4-heavy-large-${{ matrix.os }}
          if-no-files-found: ignore
          path: |
            mshqc_heavy_large_running_log.txt
            psi4_heavy_large_running_log.txt
            compare_heavy_large_log.txt
            mshqc_heavy_large_raw.json
            psi4_heavy_large_raw.json
            test_heavy_large_validation.txt
"""
    path.write_text(yml, encoding="utf-8")
    print(f"[PATCH] {path} berhasil diperbarui.")


def disable_test_geom_only_auto_trigger():
    path = ROOT / ".github" / "workflows" / "test_geom_only.yml"
    if not path.is_file():
        print(f"[SKIP] Tidak ditemukan: {path}")
        return
    txt = path.read_text(encoding="utf-8")
    # Ganti seluruh blok 'on: ...' sebelum 'jobs:' menjadi manual-only (workflow_dispatch)
    new_txt, count = re.subn(
        r"(?ms)^on:\n.*?(?=^jobs:)",
        "on:\n  workflow_dispatch:\n\n",
        txt,
        count=1,
    )
    if count == 1:
        path.write_text(new_txt, encoding="utf-8")
        print(f"[PATCH] {path} -> auto-trigger push dinonaktifkan (hanya workflow_dispatch).")
    else:
        print(f"[WARN] Pola blok 'on:' tidak cocok di {path}, cek manual.")


if __name__ == "__main__":
    patch_calculators()
    patch_heavy_large_workflow()
    disable_test_geom_only_auto_trigger()
    print("\nSelesai. Jalankan 'git diff' untuk verifikasi sebelum commit & push.")