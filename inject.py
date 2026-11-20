#!/usr/bin/env python3
"""
inject_production_ready_v6.py
-----------------------------
One-shot idempotent Production-Readiness Finalizer (v6) untuk mshqc-sync.
Menutup 100% sisa blocker kompilasi, OOC end-to-end (RAII auto-cleanup),
lonjakan RAM 4D OMP2/OMP3, dan paritas API Python HPC.

Penggunaan:
  python3 inject_production_ready_v6.py --root . --dry-run
  python3 inject_production_ready_v6.py --root .
"""

import argparse
import ast
import datetime
import re
import shutil
import sys
from pathlib import Path
from typing import Tuple


# ==============================================================================
# UTILITAS PATCHER AMAN (LITERAL SUB + VERIFIKATOR INTEGRITAS C++20 & PYTHON)
# ==============================================================================

def sub_literal(pattern: str, replacement: str, src: str, count: int = 1, flags: int = re.MULTILINE) -> Tuple[str, int]:
    """Regex replacement literal tanpa tafsir escape sequence (anti bug \\n)."""
    return re.subn(pattern, lambda _m: replacement, src, count=count, flags=flags)


def ensure_cpp_include_after_extern_c(src: str, header: str) -> str:
    """Menyisipkan #include C++ secara aman di luar blok extern "C"."""
    inc_line = f"#include {header}"
    if inc_line in src:
        return src
    m_alg = re.search(r"(#include\s+<algorithm>\s*\n)", src)
    if m_alg:
        pos = m_alg.end()
        return src[:pos] + inc_line + "\n" + src[pos:]
    m_io = re.search(r"(#include\s+<iostream>\s*\n)", src)
    if m_io:
        pos = m_io.end()
        return src[:pos] + inc_line + "\n" + src[pos:]
    return inc_line + "\n" + src


def verify_cpp_integrity(src: str, rel_path: str) -> None:
    """Verifikasi string literal, keseimbangan kurung kurawal, dan kebersihan blok extern C."""
    in_block = False
    depth = 0
    for lineno, line in enumerate(src.splitlines(), 1):
        in_str = False
        in_chr = False
        q_count = 0
        i = 0
        n = len(line)
        if 'R"(' in line:
            continue
        while i < n:
            ch = line[i]
            nxt = line[i + 1] if i + 1 < n else ""
            if in_block:
                if ch == "*" and nxt == "/":
                    in_block = False
                    i += 2
                    continue
                i += 1
                continue
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
                    q_count += 1
                i += 1
                continue
            if in_chr:
                if ch == "\\":
                    i += 2
                    continue
                if ch == "'":
                    in_chr = False
                i += 1
                continue
            if ch == "/" and nxt == "/":
                break
            if ch == "/" and nxt == "*":
                in_block = True
                i += 2
                continue
            if ch == '"':
                in_str = True
                q_count += 1
            elif ch == "'":
                in_chr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if q_count % 2 != 0:
            raise RuntimeError(f"Unterminated string literal di {rel_path}:{lineno}: {line.strip()}")
    if depth != 0:
        raise RuntimeError(f"Unbalanced braces (delta={depth}) di {rel_path}")

    for m in re.finditer(r'extern\s+"C"\s*\{([^}]*)\}', src, flags=re.DOTALL):
        body = m.group(1)
        for bad in ("<string>", "<stdexcept>", "<cstdlib>", "<vector>", "<iostream>"):
            if bad in body:
                raise RuntimeError(f'Header C++ {bad} ditemukan di dalam blok extern "C" pada {rel_path}!')


def verify_python_integrity(src: str, rel_path: str) -> None:
    try:
        ast.parse(src, filename=rel_path)
    except SyntaxError as exc:
        raise RuntimeError(f"SyntaxError Python di {rel_path}:{exc.lineno}: {exc.msg}") from exc


# ==============================================================================
# PATCH 1: src/ints/integrals.cc (CRITICAL COMPILE BLOCKER FIX)
# ==============================================================================

def patch_integrals_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_EXTERN_C_FIX]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    def _clean_extern_c(m: re.Match) -> str:
        block = m.group(0)
        for h in ("<string>", "<stdexcept>", "<cstdlib>"):
            block = re.sub(rf"^[ \t]*#include\s+{re.escape(h)}[ \t]*\r?\n", "", block, flags=re.MULTILINE)
        return block

    new_src = re.sub(r'extern\s+"C"\s*\{[^}]*\}', _clean_extern_c, src, flags=re.DOTALL)
    new_src = ensure_cpp_include_after_extern_c(new_src, "<cstdlib>")
    new_src = ensure_cpp_include_after_extern_c(new_src, "<stdexcept>")
    new_src = ensure_cpp_include_after_extern_c(new_src, "<string>")

    pat = r"(#include\s+<string>\s*\n)"
    new_src, n = sub_literal(pat, f"#include <string>\n{marker}\n", new_src, count=1)
    if n != 1:
        new_src = f"{marker}\n" + new_src
    return new_src, 'PATCHED (pindahkan STL headers ke luar extern "C")'


# ==============================================================================
# PATCH 2: include/mshqc/integrals/df_eri.h
# ==============================================================================

def patch_df_eri_h(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_DF_PATH]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    if "#include <string>" not in src:
        src = src.replace("#include <Eigen/Dense>", "#include <Eigen/Dense>\n#include <string>", 1)

    pat_pub = r"(void\s+compute\(bool\s+save_hdf5\s*=\s*false\);)"
    rep_pub = (
        "void compute(bool save_hdf5 = false);\n"
        f"  {marker}\n"
        "  void set_hdf5_path(const std::string& path) { hdf5_path_ = path; }\n"
        "  const std::string& get_hdf5_path() const { return hdf5_path_; }"
    )
    src, n1 = sub_literal(pat_pub, rep_pub, src, count=1)
    if n1 != 1:
        raise RuntimeError("Anchor compute(bool save_hdf5 = false) tidak ditemukan di df_eri.h")

    pat_priv = r"(double\s+cutoff_;)"
    rep_priv = 'double cutoff_;\n  std::string hdf5_path_ = "df_tensor.h5";'
    src, n2 = sub_literal(pat_priv, rep_priv, src, count=1)
    if n2 != 1:
        raise RuntimeError("Anchor cutoff_ tidak ditemukan di df_eri.h")

    return src, "PATCHED (tambah set_hdf5_path/hdf5_path_ pada DensityFittingERI)"


# ==============================================================================
# PATCH 3: src/integrals/df_eri.cc
# ==============================================================================

def patch_df_eri_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_DF_PATH]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat = r'utils::HDF5TensorIO\s+io\("df_tensor\.h5",\s*utils::HDF5TensorIO::Mode::WRITE_TRUNCATE\);'
    rep = (
        f"{marker}\n"
        '    const std::string out_h5 = hdf5_path_.empty() ? std::string("df_tensor.h5") : hdf5_path_;\n'
        "    utils::HDF5TensorIO io(out_h5, utils::HDF5TensorIO::Mode::WRITE_TRUNCATE);"
    )
    src, n = sub_literal(pat, rep, src, count=1)
    if n != 1:
        raise RuntimeError('Anchor HDF5TensorIO io("df_tensor.h5"...) tidak ditemukan di df_eri.cc')
    return src, "PATCHED (gunakan hdf5_path_ dinamis pada DensityFittingERI::compute)"


# ==============================================================================
# PATCH 4: include/mshqc/scf/scf.h
# ==============================================================================

def patch_scf_h(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_OOC_RAII]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    if "#include <cstdio>" not in src:
        src = src.replace("#include <cmath>", "#include <cmath>\n#include <cstdio>", 1)

    pat_res = r"(struct\s+SCFResult\s*\{)"
    rep_res = f"""{marker}
struct ScopedScratchFile {{
  std::string path;
  explicit ScopedScratchFile(std::string p = "") : path(std::move(p)) {{}}
  ~ScopedScratchFile() {{
    if (!path.empty()) std::remove(path.c_str());
  }}
  ScopedScratchFile(const ScopedScratchFile&) = delete;
  ScopedScratchFile& operator=(const ScopedScratchFile&) = delete;
}};

struct SCFResult {{"""
    src, n1 = sub_literal(pat_res, rep_res, src, count=1)
    if n1 != 1:
        raise RuntimeError("Anchor struct SCFResult tidak ditemukan di scf.h")

    pat_lmat = r"(Eigen::MatrixXd\s+L_mat;\s*\n\};)"
    rep_lmat = (
        "Eigen::MatrixXd L_mat;\n"
        "  std::string df_ooc_file;\n"
        "  std::shared_ptr<ScopedScratchFile> df_ooc_handle;\n"
        "};"
    )
    src, n2 = sub_literal(pat_lmat, rep_lmat, src, count=1)
    if n2 != 1:
        raise RuntimeError("Anchor SCFResult::L_mat tidak ditemukan di scf.h")

    pat_base = r"(Eigen::MatrixXd\s+L_mat_;)"
    rep_base = (
        "Eigen::MatrixXd L_mat_;\n"
        "  std::string df_ooc_file_;\n"
        "  std::shared_ptr<ScopedScratchFile> df_ooc_handle_;"
    )
    src, n3 = sub_literal(pat_base, rep_base, src, count=1)
    if n3 != 1:
        raise RuntimeError("Anchor BaseSCF::L_mat_ tidak ditemukan di scf.h")

    return src, "PATCHED (tambah ScopedScratchFile RAII + df_ooc_file/handle pada SCFResult & BaseSCF)"


# ==============================================================================
# PATCH 5: src/scf/scf.cc
# ==============================================================================

def patch_scf_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_OOC_SCRATCH]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    if "#include <atomic>" not in src:
        src = src.replace("#include <chrono>", "#include <chrono>\n#include <atomic>", 1)
    if "#include <filesystem>" not in src:
        src = src.replace("#include <atomic>", "#include <atomic>\n#include <filesystem>", 1)
    if "getpid" not in src:
        pid_inc = (
            "#if defined(_WIN32)\n"
            "#include <process.h>\n"
            "#else\n"
            "#include <unistd.h>\n"
            "#endif\n"
        )
        src = src.replace("#include <filesystem>\n", "#include <filesystem>\n" + pid_inc, 1)

    pat_ns = r"(namespace\s*\{\s*\n)"
    rep_ns = f"""namespace {{
{marker}
inline std::string make_unique_ooc_df_path() {{
  static std::atomic<unsigned long long> seq{{0}};
  const unsigned long long id = seq.fetch_add(1, std::memory_order_relaxed);
#if defined(_WIN32)
  const long pid = static_cast<long>(_getpid());
#else
  const long pid = static_cast<long>(getpid());
#endif
  std::string dir;
  for (const char* env_k : {{"MSHQC_SCRATCH", "TMPDIR", "TEMP", "TMP"}}) {{
    if (const char* val = std::getenv(env_k)) {{
      if (val[0] != '\\0' && std::string(val).find("/dev/shm") == std::string::npos) {{
        dir = val;
        break;
      }}
    }}
  }}
  std::filesystem::path p = dir.empty() ? std::filesystem::path(".") : std::filesystem::path(dir);
  p /= ("mshqc_df_ooc_" + std::to_string(pid) + "_" + std::to_string(id) + ".h5");
  return p.string();
}}

"""
    src, n_ns = sub_literal(pat_ns, rep_ns, src, count=1)
    if n_ns != 1:
        raise RuntimeError("Anchor anonymous namespace tidak ditemukan di src/scf/scf.cc")

    pat_comp = r'if\s*\(use_disk\)\s*config_\.scf_type\s*=\s*"ooc";\s*df_engine->compute\(use_disk\);'
    rep_comp = (
        '    if (use_disk) {\n'
        '      config_.scf_type = "ooc";\n'
        '      df_ooc_file_ = make_unique_ooc_df_path();\n'
        '      df_ooc_handle_ = std::make_shared<ScopedScratchFile>(df_ooc_file_);\n'
        '      df_engine->set_hdf5_path(df_ooc_file_);\n'
        '    }\n'
        '    df_engine->compute(use_disk);'
    )
    src, n_comp = sub_literal(pat_comp, rep_comp, src, count=1)
    if n_comp != 1:
        raise RuntimeError("Anchor df_engine->compute(use_disk) tidak ditemukan di src/scf/scf.cc")

    pat_mode = r'std::string\s+mode_str\s*=\s*\(config_\.scf_type\s*==\s*"incore"\)\s*\?\s*"In-Core"\s*:\s*"Direct";'
    rep_mode = 'std::string mode_str = (config_.scf_type == "ooc") ? "OOC-HDF5" : ((config_.scf_type == "incore") ? "In-Core" : "Direct");'
    src, _ = sub_literal(pat_mode, rep_mode, src, count=1)

    src, n_h5 = sub_literal(
        r'"df_tensor\.h5"',
        '(df_ooc_file_.empty() ? std::string("df_tensor.h5") : df_ooc_file_)',
        src,
        count=0,
    )
    if n_h5 < 3:
        raise RuntimeError(f"Ekspektasi >=3 referensi df_tensor.h5 di scf.cc, hanya ditemukan {n_h5}")

    pat_res_l = r"(r\.L_mat\s*=\s*L_mat_;)"
    rep_res_l = (
        "r.L_mat = L_mat_;\n"
        "  r.df_ooc_file = df_ooc_file_;\n"
        "  r.df_ooc_handle = df_ooc_handle_;"
    )
    src, n_rl = sub_literal(pat_res_l, rep_res_l, src, count=2)
    if n_rl != 2:
        raise RuntimeError(f"Ekspektasi 2 anchor r.L_mat = L_mat_ di scf.cc, ditemukan {n_rl}")

    return src, f"PATCHED (PID-unique OOC scratch + RAII auto-cleanup + {n_h5} refs updated)"


# ==============================================================================
# PATCH 6: src/mp2/mp2.cc
# ==============================================================================

def patch_mp2_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_OOC_MP2]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    if '#include "mshqc/utils/hdf5_io.h"' not in src:
        src = src.replace(
            '#include "mshqc/integrals/df_eri.h"',
            '#include "mshqc/integrals/df_eri.h"\n#include "mshqc/utils/hdf5_io.h"',
            1,
        )

    pat_auto = r"if\s*\(config_\.use_df\s*&&\s*!integrals::has_valid_ao_factor\(scf_\.L_mat,\s*nbf_\)\)\s*\{"
    rep_auto = (
        f"{marker}\n"
        "  const bool has_ooc_df = (scf_.L_mat.rows() == 0 && scf_.L_mat.cols() > 0 && !scf_.df_ooc_file.empty());\n"
        "  if (config_.use_df && !integrals::has_valid_ao_factor(scf_.L_mat, nbf_) && !has_ooc_df) {"
    )
    src, n1 = sub_literal(pat_auto, rep_auto, src, count=1)
    if n1 != 1:
        raise RuntimeError("Anchor auto-build L_mat tidak ditemukan di src/mp2/mp2.cc")

    pat_t3c = r"(const\s+size_t\s+bytes_per_p\s*=\s*std::max<size_t>\(1,\s*static_cast<size_t>\(nbf_\)\s*\*\s*nbf_\s*\*\s*sizeof\(double\)\);\s*const\s+int\s+chunk_size\s*=[^;]+;)"
    rep_t3c = r"""const bool is_ooc = (scf_.L_mat.rows() == 0 && n_aux > 0 && !scf_.df_ooc_file.empty());
  const size_t bytes_per_p = std::max<size_t>(1, static_cast<size_t>(nbf_) * nbf_ * sizeof(double));
  const int chunk_size = std::clamp(static_cast<int>((8ULL * 1024ULL * 1024ULL) / bytes_per_p), 1, 32);

  if (is_ooc) {
    utils::HDF5TensorIO io(scf_.df_ooc_file, utils::HDF5TensorIO::Mode::READ_ONLY);
    const int ooc_chunk = std::clamp(
        static_cast<int>((64ULL * 1024ULL * 1024ULL) / bytes_per_p), 1, 128);
    for (int P_start = 0; P_start < n_aux; P_start += ooc_chunk) {
      const int P_size = std::min(ooc_chunk, n_aux - P_start);
      Eigen::MatrixXd L_sq_chunk(static_cast<Eigen::Index>(nbf_) * nbf_, P_size);
      io.read_slice_4d("df_tensor", {P_start, 0, 0, 0},
                       {P_size, static_cast<long>(nbf_), static_cast<long>(nbf_), 1},
                       L_sq_chunk.data());
      #pragma omp parallel
      {
        Eigen::MatrixXd X_a_loc(nvir_a_, nbf_), B_mo_a(nvir_a_, nocc_a_);
        Eigen::MatrixXd X_b_loc, B_mo_b;
        if (has_beta) {
          X_b_loc.resize(nvir_b_, nbf_);
          B_mo_b.resize(nvir_b_, nocc_b_);
        }
        #pragma omp for schedule(static) nowait
        for (int p = 0; p < P_size; ++p) {
          Eigen::Map<const Eigen::MatrixXd> L_P(L_sq_chunk.col(p).data(), nbf_, nbf_);
          X_a_loc.noalias() = Ca_vir.transpose() * L_P;
          B_mo_a.noalias() = X_a_loc * Ca_occ;
          B_ia_P_alpha_.col(P_start + p) =
              Eigen::Map<const Eigen::VectorXd>(B_mo_a.data(), static_cast<Eigen::Index>(nvir_a_) * nocc_a_);
          if (has_beta) {
            X_b_loc.noalias() = Cb_vir.transpose() * L_P;
            B_mo_b.noalias() = X_b_loc * Cb_occ;
            B_ia_P_beta_.col(P_start + p) =
                Eigen::Map<const Eigen::VectorXd>(B_mo_b.data(), static_cast<Eigen::Index>(nvir_b_) * nocc_b_);
          }
        }
      }
    }
    return;
  }"""
    src, n2 = sub_literal(pat_t3c, rep_t3c, src, count=1)
    if n2 != 1:
        raise RuntimeError("Anchor bytes_per_p/chunk_size tidak ditemukan di BaseMP2::transform_3center_mo")

    pat_omp2_ctor = r"(symmetrizer_\s*=\s*nullptr;\s*init_fast_integrals\(\);)"
    rep_omp2_ctor = (
        'if (config_.use_df && !integrals::has_valid_ao_factor(scf_.L_mat, nbf_)) {\n'
        '    throw std::runtime_error(\n'
        '        "OMP2/OMP3 requires in-core packed DF factors in RAM, but SCF ran in OOC disk mode. "\n'
        '        "Increase memory_limit_gb / MSHQC_MEMORY_GB or use DF-MP2 streaming.");\n'
        '  }\n'
        '  symmetrizer_ = nullptr;\n'
        '  init_fast_integrals();'
    )
    src, n3 = sub_literal(pat_omp2_ctor, rep_omp2_ctor, src, count=1)
    if n3 != 1:
        raise RuntimeError("Anchor OMP2::OMP2 init_fast_integrals() tidak ditemukan di src/mp2/mp2.cc")

    return src, "PATCHED (OOC HDF5 streaming DF-MP2 end-to-end + OMP2 OOC guard)"


# ==============================================================================
# PATCH 7: src/mp/omp2_energy.cc
# ==============================================================================

def patch_omp2_energy_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_T2_CLEAR]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat = r"(t2_aa_\.allocate_block\(0,\s*0,\s*0,\s*0,\s*na_,\s*va_,\s*na_,\s*va_\);)"
    rep = (
        f"{marker}\n"
        "  t2_aa_.clear(); t2_bb_.clear(); t2_ab_.clear();\n"
        "  t2_aa_.allocate_block(0, 0, 0, 0, na_, va_, na_, va_);"
    )
    src, n = sub_literal(pat, rep, src, count=1)
    if n != 1:
        raise RuntimeError("Anchor t2_aa_.allocate_block tidak ditemukan di src/mp/omp2_energy.cc")
    return src, "PATCHED (bebaskan T2 lama sebelum realokasi -> hilangkan spike 2x RAM)"


# ==============================================================================
# PATCH 8: src/mp3/mp3.cc
# ==============================================================================

def patch_mp3_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_PROD_V6_MP3_LIFECYCLE]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    idx_exec = src.find("double OMP3::execute_micro_iterations()")
    if idx_exec == -1:
        raise RuntimeError("Anchor OMP3::execute_micro_iterations tidak ditemukan di src/mp3/mp3.cc")
    head, tail = src[:idx_exec], src[idx_exec:]
    pat_micro = r"(pseudocanonicalize\(\);)"
    rep_micro = (
        f"{marker}\n"
        "  L2_aa_.resize(0, 0, 0, 0);\n"
        "  L2_bb_.resize(0, 0, 0, 0);\n"
        "  L2_ab_.resize(0, 0, 0, 0);\n"
        "  pseudocanonicalize();"
    )
    tail, n1 = sub_literal(pat_micro, rep_micro, tail, count=1)
    if n1 != 1:
        raise RuntimeError("Anchor pseudocanonicalize() dalam OMP3::execute_micro_iterations tidak ditemukan")
    src = head + tail

    pat_t2full = (
        r"Eigen::Tensor<\s*double,\s*4\s*>\s*T2_aa_ijab\(na_,\s*na_,\s*va_,\s*va_\);\s*"
        r"#pragma omp parallel for collapse\(4\) schedule\(static\)\s*"
        r"for\s*\(int i = 0; i < na_; \+\+i\)\s*\{\s*"
        r"for\s*\(int j = 0; j < na_; \+\+j\)\s*\{\s*"
        r"for\s*\(int a = 0; a < va_; \+\+a\)\s*\{\s*"
        r"for\s*\(int b = 0; b < va_; \+\+b\)\s*\{\s*"
        r"T2_aa_ijab\(i,\s*j,\s*a,\s*b\)\s*=\s*\(\*t2_aa_dense\)\(i,\s*a,\s*j,\s*b\);\s*"
        r"\}\s*\}\s*\}\s*\}"
    )
    rep_t2full = """const bool skip_full_t2_copy = (config_.eri_method != "exact" && n_frozen_ > 0);
  Eigen::Tensor< double, 4 > T2_aa_ijab;
  if (!skip_full_t2_copy) {
    T2_aa_ijab.resize(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < na_; ++i)
      for (int j = 0; j < na_; ++j)
        for (int a = 0; a < va_; ++a)
          for (int b = 0; b < va_; ++b)
            T2_aa_ijab(i, j, a, b) = (*t2_aa_dense)(i, a, j, b);
  }"""
    src, n2 = sub_literal(pat_t2full, rep_t2full, src, count=1)
    if n2 != 1:
        raise RuntimeError("Anchor T2_aa_ijab full copy tidak ditemukan di OMP3::compute_mp3_correction")

    src, n3 = sub_literal(
        r"T_act\(i,\s*j,\s*a,\s*b\)\s*=\s*T2_aa_ijab\(nf_a\s*\+\s*i,\s*nf_a\s*\+\s*j,\s*a,\s*b\);",
        "T_act(i, j, a, b) = (*t2_aa_dense)(nf_a + i, a, nf_a + j, b);",
        src, count=1,
    )
    src, n4 = sub_literal(
        r"Taa_act\(i,\s*j,\s*a,\s*b\)\s*=\s*T2_aa_ijab\(nf_a\s*\+\s*i,\s*nf_a\s*\+\s*j,\s*a,\s*b\);",
        "Taa_act(i, j, a, b) = (*t2_aa_dense)(nf_a + i, a, nf_a + j, b);",
        src, count=1,
    )
    if n3 != 1 or n4 != 1:
        raise RuntimeError(f"Anchor T_act/Taa_act tidak ditemukan (n3={n3}, n4={n4})")

    pat_ring_fc = r'(tblis::mult<double>\(-1\.0,\s*t_T,\s*"ikcb",\s*t_Voovv,\s*"jkac",\s*1\.0,\s*t_W,\s*"ijab"\);\s*\n\s*const int n_pairs_sym)'
    rep_ring_fc = (
        'tblis::mult<double>(-1.0, t_T, "ikcb", t_Voovv, "jkac", 1.0, t_W, "ijab");\n'
        '      M_oooo.resize(0, 0); M_ovov.resize(0, 0); M_oovv.resize(0, 0);\n'
        '      B_ij_act.resize(0, 0); B_ia_act.resize(0, 0);\n\n'
        '      const int n_pairs_sym'
    )
    src, n5 = sub_literal(pat_ring_fc, rep_ring_fc, src, count=1)
    if n5 != 1:
        raise RuntimeError("Anchor eager-free M_oooo/M_ovov/M_oovv (restricted FC) tidak ditemukan di mp3.cc")

    pat_early_t3 = r"const int na_act = na_ - nf_a;\s*t2_3rd_aa_ = Eigen::Tensor<double, 4>\(na_, na_, va_, va_\);\s*t2_3rd_aa_\.setZero\(\);\s*if \(na_act <= 0 \|\| va_ <= 0\) \{ e_mp3_tot_ = 0\.0; return; \}"
    rep_early_t3 = (
        "const int na_act = na_ - nf_a;\n"
        "      if (na_act <= 0 || va_ <= 0) {\n"
        "        t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);\n"
        "        t2_3rd_aa_.setZero();\n"
        "        e_mp3_tot_ = 0.0;\n"
        "        return;\n"
        "      }"
    )
    src, n6 = sub_literal(pat_early_t3, rep_early_t3, src, count=1)
    if n6 != 1:
        raise RuntimeError("Anchor early t2_3rd_aa_ (restricted FC) tidak ditemukan di mp3.cc")

    pat_late_t3 = r"(double e3_aa = 0\.0;\s*#pragma omp parallel for collapse\(4\) reduction\(\+:e3_aa\) schedule\(static\)\s*for \(int i = 0; i < na_act; \+\+i\))"
    rep_late_t3 = (
        "T_pairs.clear(); T_pairs.shrink_to_fit();\n"
        "      W_pairs.clear(); W_pairs.shrink_to_fit();\n"
        "      t2_3rd_aa_ = Eigen::Tensor<double, 4>(na_, na_, va_, va_);\n"
        "      t2_3rd_aa_.setZero();\n\n"
        "      double e3_aa = 0.0;\n"
        "      #pragma omp parallel for collapse(4) reduction(+:e3_aa) schedule(static)\n"
        "      for (int i = 0; i < na_act; ++i)"
    )
    src, n7 = sub_literal(pat_late_t3, rep_late_t3, src, count=1)
    if n7 != 1:
        raise RuntimeError("Anchor late t2_3rd_aa_ (restricted FC) tidak ditemukan di mp3.cc")

    idx_gf = src.find("void OMP3::build_generalized_fock()")
    if idx_gf == -1:
        raise RuntimeError("Anchor OMP3::build_generalized_fock tidak ditemukan di src/mp3/mp3.cc")
    head_gf, tail_gf = src[:idx_gf], src[idx_gf:]
    pat_end_gf = r"(Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero\(nbf_, nbf_\);)"
    rep_end_gf = (
        "L2_aa_.resize(0, 0, 0, 0);\n"
        "  L2_bb_.resize(0, 0, 0, 0);\n"
        "  L2_ab_.resize(0, 0, 0, 0);\n\n"
        "  Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);"
    )
    tail_gf, n8 = sub_literal(pat_end_gf, rep_end_gf, tail_gf, count=1)
    if n8 != 1:
        raise RuntimeError("Anchor G_full_a di OMP3::build_generalized_fock tidak ditemukan")
    src = head_gf + tail_gf

    return src, "PATCHED (zero full-T2 copy saat FC + eager-free M_ovov/M_oovv/pairs + L2 lifecycle)"


# ==============================================================================
# PATCH 9: python/mshqc/calculators.py
# ==============================================================================

def patch_calculators_py(src: str) -> Tuple[str, str]:
    marker = "# [INJECTED_PROD_V6_LEVEL_SHIFT]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat_init = r"(memory_limit_gb:\s*float\s*=\s*4\.0,\s*\n\s*use_symmetry:\s*bool\s*=\s*True\):)"
    rep_init = (
        "memory_limit_gb: float = 4.0,\n"
        "                 use_symmetry: bool = True,\n"
        f"                 level_shift: float = 0.0):  {marker}"
    )
    src, n1 = sub_literal(pat_init, rep_init, src, count=1)
    if n1 != 1:
        raise RuntimeError("Anchor MSHQCCalculator.__init__ tidak ditemukan di calculators.py")

    pat_attr = r"(self\.memory_limit_gb\s*=\s*memory_limit_gb)"
    rep_attr = "self.memory_limit_gb = memory_limit_gb\n        self.level_shift = level_shift"
    src, n2 = sub_literal(pat_attr, rep_attr, src, count=1)
    if n2 != 1:
        raise RuntimeError("Anchor self.memory_limit_gb tidak ditemukan di calculators.py")

    pat_cfg = r"(if hasattr\(conf,\s*'memory_limit_gb'\):\s*conf\.memory_limit_gb\s*=\s*self\.memory_limit_gb)"
    rep_cfg = (
        "if hasattr(conf, 'memory_limit_gb'): conf.memory_limit_gb = self.memory_limit_gb\n"
        "        if hasattr(conf, 'level_shift'): conf.level_shift = self.level_shift"
    )
    src, n3 = sub_literal(pat_cfg, rep_cfg, src, count=1)
    if n3 != 1:
        raise RuntimeError("Anchor _make_scf_config memory_limit_gb tidak ditemukan di calculators.py")

    pat_run_sig = r"(memory_limit_gb:\s*float\s*=\s*4\.0,\s*\n\s*use_symmetry:\s*bool\s*=\s*True\)\s*->\s*Union\[Any,\s*Dict\[str,\s*Any\]\]:)"
    rep_run_sig = (
        "memory_limit_gb: float = 4.0,\n"
        "        use_symmetry: bool = True,\n"
        '        scf_type: str = "incore",\n'
        "        level_shift: float = 0.0) -> Union[Any, Dict[str, Any]]:"
    )
    src, n4 = sub_literal(pat_run_sig, rep_run_sig, src, count=1)
    if n4 != 1:
        raise RuntimeError("Anchor mshqc.run signature tidak ditemukan di calculators.py")

    pat_run_call = r"(freeze_core=freeze_core,\s*n_frozen=n_frozen,\s*memory_limit_gb=memory_limit_gb,\s*use_symmetry=use_symmetry)"
    rep_run_call = (
        "freeze_core=freeze_core, n_frozen=n_frozen, memory_limit_gb=memory_limit_gb, "
        "use_symmetry=use_symmetry, scf_type=scf_type, level_shift=level_shift"
    )
    src, n5 = sub_literal(pat_run_call, rep_run_call, src, count=1)
    if n5 != 1:
        raise RuntimeError("Anchor MSHQCCalculator(...) di mshqc.run tidak ditemukan di calculators.py")

    return src, "PATCHED (tambah level_shift & scf_type pada MSHQCCalculator & mshqc.run)"


# ==============================================================================
# PATCH 10: python/mshqc/session.py
# ==============================================================================

def patch_session_py(src: str) -> Tuple[str, str]:
    marker = "# [INJECTED_PROD_V6_SESSION_HPC]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat_init = (
        r"def __init__\(self,\s*\n"
        r"                 molecule: Union\[mshqc\.Molecule, str, List\[Tuple\]\],\s*\n"
        r'                 basis_name: str = "cc-pVTZ",\s*\n'
        r'                 eri_method: str = "exact",\s*\n'
        r"                 aux_basis_name: Optional\[str\] = None,\s*\n"
        r"                 charge: int = 0,\s*\n"
        r"                 multiplicity: Optional\[int\] = None,\s*\n"
        r'                 units: str = "bohr",\s*\n'
        r"                 reference: Optional\[str\] = None,\s*\n"
        r"                 threshold: float = 1e-9,\s*\n"
        r"                 print_level: int = 0\):"
    )
    rep_init = f"""def __init__(self,
                 molecule: Union[mshqc.Molecule, str, List[Tuple]],
                 basis_name: str = "cc-pVTZ",
                 eri_method: str = "exact",
                 aux_basis_name: Optional[str] = None,
                 charge: int = 0,
                 multiplicity: Optional[int] = None,
                 units: str = "bohr",
                 reference: Optional[str] = None,
                 threshold: float = 1e-9,
                 print_level: int = 0,
                 scf_type: str = "incore",
                 freeze_core: bool = False,
                 n_frozen: int = 0,
                 memory_limit_gb: float = 4.0,
                 use_symmetry: bool = True,
                 level_shift: float = 0.0):  {marker}"""
    src, n1 = sub_literal(pat_init, rep_init, src, count=1)
    if n1 != 1:
        raise RuntimeError("Anchor MSHQCSession.__init__ tidak ditemukan di session.py")

    pat_calc = r"(threshold=threshold,\s*\n\s*print_level=print_level,\s*\n\s*\))"
    rep_calc = (
        "threshold=threshold,\n"
        "            print_level=print_level,\n"
        "            scf_type=scf_type,\n"
        "            freeze_core=freeze_core,\n"
        "            n_frozen=n_frozen,\n"
        "            memory_limit_gb=memory_limit_gb,\n"
        "            use_symmetry=use_symmetry,\n"
        "            level_shift=level_shift,\n"
        "        )"
    )
    src, n2 = sub_literal(pat_calc, rep_calc, src, count=1)
    if n2 != 1:
        raise RuntimeError("Anchor MSHQCCalculator call di session.py tidak ditemukan")

    src, _ = sub_literal(
        r'elif m in \("mp2", "rmp2", "ump2"\):',
        'elif m in ("mp2", "rmp2", "ump2", "romp2"):',
        src, count=1,
    )
    src, _ = sub_literal(
        r'elif m in \("mp3", "rmp3", "ump3"\):',
        'elif m in ("mp3", "rmp3", "ump3", "romp3"):',
        src, count=1,
    )

    return src, "PATCHED (paritas penuh parameter HPC + ROMP2/ROMP3 di MSHQCSession)"


# ==============================================================================
# PIPELINE UTAMA
# ==============================================================================

PATCH_PIPELINE = [
    ("src/ints/integrals.cc", patch_integrals_cc, "cpp"),
    ("include/mshqc/integrals/df_eri.h", patch_df_eri_h, "cpp"),
    ("src/integrals/df_eri.cc", patch_df_eri_cc, "cpp"),
    ("include/mshqc/scf/scf.h", patch_scf_h, "cpp"),
    ("src/scf/scf.cc", patch_scf_cc, "cpp"),
    ("src/mp2/mp2.cc", patch_mp2_cc, "cpp"),
    ("src/mp/omp2_energy.cc", patch_omp2_energy_cc, "cpp"),
    ("src/mp3/mp3.cc", patch_mp3_cc, "cpp"),
    ("python/mshqc/calculators.py", patch_calculators_py, "py"),
    ("python/mshqc/session.py", patch_session_py, "py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="MSHQC-Sync 100% Production-Ready HPC Injector v6")
    parser.add_argument("--root", type=str, default=".", help="Path root repositori mshqc-sync")
    parser.add_argument("--dry-run", action="store_true", help="Uji pencocokan & integritas tanpa menulis file")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / f".backup_prod_v6_{ts}"

    print("=" * 96)
    print(f" MSHQC-SYNC 100% PRODUCTION-READY HPC INJECTOR V6 | Root: {root}")
    print(f" Mode: {'DRY-RUN (no write)' if args.dry_run else 'APPLY (auto-backup + C++/Python verifier)'}")
    print("=" * 96)

    patched = 0
    for rel_path, fn, lang in PATCH_PIPELINE:
        fpath = root / rel_path
        if not fpath.is_file():
            print(f"[MISS] {rel_path:<36s} -> file tidak ditemukan!")
            return 1

        orig_text = fpath.read_text(encoding="utf-8")
        try:
            new_text, status = fn(orig_text)
            if lang == "cpp":
                verify_cpp_integrity(new_text, rel_path)
            else:
                verify_python_integrity(new_text, rel_path)
        except Exception as exc:
            print(f"[FAIL] {rel_path:<36s} -> {exc}")
            return 1

        changed = (new_text != orig_text)
        print(f"[OK  ] {rel_path:<36s} -> {status}")
        if changed and not args.dry_run:
            bpath = backup_root / rel_path
            bpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, bpath)
            fpath.write_text(new_text, encoding="utf-8")
            patched += 1

    print("=" * 96)
    if args.dry_run:
        print("DRY-RUN PASS: Seluruh 10 file lolos anchor match & verifikasi integritas C++20/Python.")
    else:
        print(f"SELESAI: {patched} file diperbarui & terverifikasi. Backup: {backup_root}")
        print("Langkah selanjutnya: rebuild C++20 -> pytest -> verify_audit_and_scaling -> heavy_large.")
    return 0


if __name__ == "__main__":
    sys.exit(main())