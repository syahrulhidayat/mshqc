#!/usr/bin/env python3
"""
inject_hpc_heavy_molecules_v5.py
--------------------------------
One-shot idempotent HPC Heavy-Molecule Production Upgrade Injector (v5) untuk mshqc-sync.
Menutup seluruh bottleneck memori/OOC/config hasil audit pasca-v4 agar mshqc-sync
siap produksi HPC untuk molekul ringan -> berat (400-800+ BF) tanpa mengubah akurasi
numerik (tetap 0.00 uHa terhadap Psi4).

Penggunaan:
  python3 inject_hpc_heavy_molecules_v5.py --root . --dry-run
  python3 inject_hpc_heavy_molecules_v5.py --root .
"""

import argparse
import datetime
import re
import shutil
import sys
from pathlib import Path
from typing import Tuple


# ==============================================================================
# UTILITAS PATCHER C++ AMAN (LITERAL SUB, BALANCED-BRACE, QUOTE VERIFIER)
# ==============================================================================

def sub_literal(pattern: str, replacement: str, src: str, count: int = 1, flags: int = re.MULTILINE) -> Tuple[str, int]:
    """Penggantian regex aman tanpa tafsir escape sequence pada replacement (anti bug \\n)."""
    return re.subn(pattern, lambda _m: replacement, src, count=count, flags=flags)


def find_cpp_function_span(src: str, signature_regex: str) -> Tuple[int, int]:
    """Mencari span [start, end) fungsi C++ lengkap hingga kurung kurawal '}' seimbang."""
    m = re.search(signature_regex, src, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Signature C++ tidak ditemukan: {signature_regex}")

    start = m.start()
    brace_open = src.find("{", m.end() - 1)
    if brace_open == -1:
        raise RuntimeError(f"Pembuka '{{' tidak ditemukan untuk: {signature_regex}")

    i = brace_open
    n = len(src)
    depth = 0
    in_str = False
    in_chr = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
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
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "'":
            in_chr = True
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1

    raise RuntimeError(f"Penutup '}}' tidak seimbang untuk: {signature_regex}")


def replace_cpp_function(src: str, signature_regex: str, replacement: str) -> str:
    start, end = find_cpp_function_span(src, signature_regex)
    prefix = src[:start]
    if prefix and not prefix.endswith("\n") and prefix.rstrip(" \t").endswith("}"):
        prefix += "\n\n"
    suffix = src[end:]
    if suffix and not suffix.startswith("\n"):
        suffix = "\n\n" + suffix
    return prefix + replacement.strip("\n") + suffix


def ensure_include(src: str, header: str) -> str:
    inc_line = f"#include {header}"
    if inc_line in src:
        return src
    m = re.search(r"(#include\s+<[^>]+>\s*\n)", src)
    if m:
        pos = m.end()
        return src[:pos] + inc_line + "\n" + src[pos:]
    return inc_line + "\n" + src


def verify_cpp_integrity(src: str, rel_path: str) -> None:
    """Memastikan tidak ada string literal terpotong (unterminated quote) atau brace tidak seimbang."""
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
            raise RuntimeError(f"Integrity check GAGAL (unterminated string) di {rel_path}:{lineno}: {line.strip()}")
    if depth != 0:
        raise RuntimeError(f"Integrity check GAGAL (unbalanced braces delta={depth}) di {rel_path}")


# ==============================================================================
# PATCH 1: include/mshqc/scf/scf.h
# ==============================================================================

def patch_scf_h(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_SCF_CONFIG]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat = r"(Eigen::MatrixXd\s+init_C_beta;)"
    rep = (
        "Eigen::MatrixXd init_C_beta;\n"
        f"  {marker}\n"
        "  double memory_limit_gb = 4.0;\n"
        "  double level_shift = 0.0;"
    )
    src, n = sub_literal(pat, rep, src, count=1)
    if n != 1:
        raise RuntimeError("Anchor init_C_beta tidak ditemukan di include/mshqc/scf/scf.h")
    return src, "PATCHED (tambah memory_limit_gb & level_shift di SCFConfig)"


# ==============================================================================
# PATCH 2: include/mshqc/mp2/mp2.h
# ==============================================================================

def patch_mp2_h(src: str) -> Tuple[str, str]:
    marker = "/* [INJECTED_HPC_V5_ACTINIDE_FC] */"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat = r"else\s+if\s*\(Z\s*<=\s*71\)\s*nf\s*\+=\s*27;\s*else\s+nf\s*\+=\s*34;"
    rep = (
        "else if (Z <= 71) nf += 27;\n"
        "    else if (Z <= 86) nf += 34;\n"
        f"    else nf += 43; {marker}"
    )
    src, n = sub_literal(pat, rep, src, count=1)
    if n != 1:
        raise RuntimeError("Anchor compute_standard_frozen_core tidak ditemukan di include/mshqc/mp2/mp2.h")
    return src, "PATCHED (frozen-core aktinida Z>=87 -> 43 orbital)"


# ==============================================================================
# PATCH 3: include/mshqc/gradient/hpc_geom_opt.h
# ==============================================================================

def patch_hpc_geom_opt_h(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_GEOM_CONFIG]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat = r'(std::string\s+xyz_traj_file\s*=\s*"mshqc_opt_traj\.xyz";)'
    rep = (
        'std::string xyz_traj_file = "mshqc_opt_traj.xyz";\n'
        f"  {marker}\n"
        "  bool freeze_core = false;\n"
        "  int n_frozen = 0;\n"
        "  double memory_limit_gb = 4.0;"
    )
    src, n = sub_literal(pat, rep, src, count=1)
    if n != 1:
        raise RuntimeError("Anchor xyz_traj_file tidak ditemukan di include/mshqc/gradient/hpc_geom_opt.h")
    return src, "PATCHED (tambah freeze_core, n_frozen, memory_limit_gb di GeomOptConfig)"


# ==============================================================================
# PATCH 4: python/bindings.cc
# ==============================================================================

def patch_bindings_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_BINDINGS]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat_z = r'\{"Tl",81\},\{"Pb",82\},\{"Bi",83\},\{"Po",84\},\{"At",85\},\{"Rn",86\}'
    rep_z = (
        '{"Tl",81},{"Pb",82},{"Bi",83},{"Po",84},{"At",85},{"Rn",86},\n'
        '            {"Fr",87},{"Ra",88},{"Ac",89},{"Th",90},{"Pa",91},{"U",92},{"Np",93},{"Pu",94},\n'
        '            {"Am",95},{"Cm",96},{"Bk",97},{"Cf",98},{"Es",99},{"Fm",100},{"Md",101},{"No",102},\n'
        '            {"Lr",103},{"Rf",104},{"Db",105},{"Sg",106},{"Bh",107},{"Hs",108},{"Mt",109},{"Ds",110},\n'
        '            {"Rg",111},{"Cn",112},{"Nh",113},{"Fl",114},{"Mc",115},{"Lv",116},{"Ts",117},{"Og",118}'
    )
    src, n_z = sub_literal(pat_z, rep_z, src, count=1)
    if n_z != 1:
        raise RuntimeError("Anchor symbol_to_z (Rn 86) tidak ditemukan di python/bindings.cc")

    pat_scf = r'\.def_rw\("aux_basis_name",\s*&SCFConfig::aux_basis_name\);'
    rep_scf = (
        '.def_rw("aux_basis_name", &SCFConfig::aux_basis_name)\n'
        f'        {marker}\n'
        '        .def_rw("memory_limit_gb", &SCFConfig::memory_limit_gb)\n'
        '        .def_rw("level_shift", &SCFConfig::level_shift);'
    )
    src, n_scf = sub_literal(pat_scf, rep_scf, src, count=1)
    if n_scf != 1:
        raise RuntimeError("Anchor SCFConfig::aux_basis_name tidak ditemukan di python/bindings.cc")

    pat_geom = r'\.def_rw\("xyz_traj_file",\s*&mshqc::gradient::GeomOptConfig::xyz_traj_file\);'
    rep_geom = (
        '.def_rw("xyz_traj_file", &mshqc::gradient::GeomOptConfig::xyz_traj_file)\n'
        '        .def_rw("freeze_core", &mshqc::gradient::GeomOptConfig::freeze_core)\n'
        '        .def_rw("n_frozen", &mshqc::gradient::GeomOptConfig::n_frozen)\n'
        '        .def_rw("memory_limit_gb", &mshqc::gradient::GeomOptConfig::memory_limit_gb);'
    )
    src, n_geom = sub_literal(pat_geom, rep_geom, src, count=1)
    if n_geom != 1:
        raise RuntimeError("Anchor GeomOptConfig::xyz_traj_file tidak ditemukan di python/bindings.cc")

    return src, "PATCHED (symbol_to_z Z=1..118 + SCFConfig & GeomOptConfig bindings)"


# ==============================================================================
# PATCH 5: src/utils/hdf5_io.cc
# ==============================================================================

def patch_hdf5_io_cc(src: str) -> Tuple[str, str]:
    marker = "/* [INJECTED_HPC_V5_FAST_HDF5] Uncompressed chunked binary I/O */"
    if "[INJECTED_HPC_V5_FAST_HDF5]" in src:
        return src, "SKIP (sudah terpasang)"

    pat = r"plist\.setDeflate\(\s*6\s*\);[^\n]*"
    src, n = sub_literal(pat, marker, src, count=1)
    if n != 1:
        raise RuntimeError("Anchor plist.setDeflate(6) tidak ditemukan di src/utils/hdf5_io.cc")
    return src, "PATCHED (hapus ZLIB deflate-6 untuk kecepatan I/O OOC)"


# ==============================================================================
# PATCH 6: src/ints/integrals.cc
# ==============================================================================

def patch_integrals_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_ERI_OOM_GUARD]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    src = ensure_include(src, "<cstdlib>")
    src = ensure_include(src, "<stdexcept>")
    src = ensure_include(src, "<string>")

    pat = r"cached_eri\.resize\(nbasis_,\s*nbasis_,\s*nbasis_,\s*nbasis_\);"
    rep = f"""{marker}
  {{
    double mem_lim_gb = 8.0;
    if (const char* env_mem = std::getenv("MSHQC_MEMORY_GB")) {{
      const double v = std::atof(env_mem);
      if (v >= 0.5) mem_lim_gb = v;
    }}
    const double est_gb =
        static_cast<double>(nbasis_) * nbasis_ * nbasis_ * nbasis_ *
        static_cast<double>(sizeof(double)) / 1.0e9;
    if (est_gb > mem_lim_gb * 0.85) {{
      throw std::runtime_error(
          "Exact 4-center ERI tensor requires " + std::to_string(est_gb) +
          " GB RAM (nbasis=" + std::to_string(nbasis_) +
          "), exceeding budget (" + std::to_string(mem_lim_gb * 0.85) +
          " GB). Use eri_method='df' or 'cholesky', or set MSHQC_MEMORY_GB.");
    }}
  }}
  cached_eri.resize(nbasis_, nbasis_, nbasis_, nbasis_);"""
    src, n = sub_literal(pat, rep, src, count=1)
    if n != 1:
        raise RuntimeError("Anchor cached_eri.resize tidak ditemukan di src/ints/integrals.cc")
    return src, "PATCHED (OOM guard sebelum alokasi Exact 4D ERI)"


# ==============================================================================
# PATCH 7: src/scf/scf.cc
# ==============================================================================

def patch_scf_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_SCF_OOC_AND_BATCH]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat_ooc = r"df_engine->compute\(use_disk\);"
    rep_ooc = (
        f"{marker}\n"
        '    if (use_disk) config_.scf_type = "ooc";\n'
        "    df_engine->compute(use_disk);"
    )
    src, n_ooc = sub_literal(pat_ooc, rep_ooc, src, count=1)
    if n_ooc != 1:
        raise RuntimeError("Anchor df_engine->compute(use_disk) tidak ditemukan di src/scf/scf.cc")

    new_single = """void compute_exchange_cholesky_single(
    const double* L_ptr,
    int nbasis,
    int k_size,
    const Eigen::MatrixXd& C_occ,
    Eigen::MatrixXd& K_acc,
    bool is_packed = false)
{
  const int n_occ = static_cast<int>(C_occ.cols());
  if (n_occ <= 0 || k_size <= 0) return;

  const int batch_p = std::clamp(
      static_cast<int>((8ULL * 1024ULL * 1024ULL) /
                       std::max<size_t>(1, static_cast<size_t>(nbasis) * n_occ * sizeof(double))),
      1, 32);
  const size_t col_stride = is_packed
      ? static_cast<size_t>(integrals::ao_tri_size(nbasis))
      : static_cast<size_t>(nbasis) * nbasis;
  const int max_threads = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> K_tls(max_threads, Eigen::MatrixXd::Zero(nbasis, nbasis));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    Eigen::MatrixXd T_batch(nbasis, n_occ * batch_p);
    Eigen::MatrixXd L_buf;
    if (is_packed) L_buf.resize(nbasis, nbasis);
    int count_in_batch = 0;

    #pragma omp for schedule(static) nowait
    for (int P = 0; P < k_size; ++P) {
      const double* col_p = L_ptr + static_cast<size_t>(P) * col_stride;
      if (is_packed) {
        integrals::unpack_ao_column(col_p, nbasis, true, L_buf);
        T_batch.middleCols(count_in_batch * n_occ, n_occ).noalias() = L_buf * C_occ;
      } else {
        Eigen::Map<const Eigen::MatrixXd> L_P(col_p, nbasis, nbasis);
        T_batch.middleCols(count_in_batch * n_occ, n_occ).noalias() = L_P * C_occ;
      }
      count_in_batch++;

      if (count_in_batch == batch_p) {
        K_tls[tid].noalias() += T_batch * T_batch.transpose();
        count_in_batch = 0;
      }
    }
    if (count_in_batch > 0) {
      auto T_rem = T_batch.leftCols(count_in_batch * n_occ);
      K_tls[tid].noalias() += T_rem * T_rem.transpose();
    }
  }

  for (int t = 0; t < max_threads; ++t) {
    K_acc.noalias() += K_tls[t];
  }
}"""

    new_dual = """void compute_exchange_cholesky_dual(
    const double* L_ptr,
    int nbasis,
    int k_size,
    const Eigen::MatrixXd& Ca_occ,
    const Eigen::MatrixXd& Cb_occ,
    Eigen::MatrixXd& Ka_acc,
    Eigen::MatrixXd& Kb_acc,
    bool is_packed = false)
{
  const int na = static_cast<int>(Ca_occ.cols());
  const int nb = static_cast<int>(Cb_occ.cols());
  if ((na + nb) <= 0 || k_size <= 0) return;
  if (nb <= 0) {
    compute_exchange_cholesky_single(L_ptr, nbasis, k_size, Ca_occ, Ka_acc, is_packed);
    return;
  }

  Eigen::MatrixXd Cab_occ(nbasis, na + nb);
  Cab_occ.leftCols(na) = Ca_occ;
  Cab_occ.rightCols(nb) = Cb_occ;

  const int batch_p = std::clamp(
      static_cast<int>((8ULL * 1024ULL * 1024ULL) /
                       std::max<size_t>(1, static_cast<size_t>(nbasis) * (na + nb) * sizeof(double))),
      1, 32);
  const size_t col_stride = is_packed
      ? static_cast<size_t>(integrals::ao_tri_size(nbasis))
      : static_cast<size_t>(nbasis) * nbasis;
  const int max_threads = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> Ka_tls(max_threads, Eigen::MatrixXd::Zero(nbasis, nbasis));
  std::vector<Eigen::MatrixXd> Kb_tls(max_threads, Eigen::MatrixXd::Zero(nbasis, nbasis));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    Eigen::MatrixXd Tab_loc(nbasis, na + nb);
    Eigen::MatrixXd Ta_batch(nbasis, na * batch_p);
    Eigen::MatrixXd Tb_batch(nbasis, nb * batch_p);
    Eigen::MatrixXd L_buf;
    if (is_packed) L_buf.resize(nbasis, nbasis);
    int count_in_batch = 0;

    #pragma omp for schedule(static) nowait
    for (int P = 0; P < k_size; ++P) {
      const double* col_p = L_ptr + static_cast<size_t>(P) * col_stride;
      if (is_packed) {
        integrals::unpack_ao_column(col_p, nbasis, true, L_buf);
        Tab_loc.noalias() = L_buf * Cab_occ;
      } else {
        Eigen::Map<const Eigen::MatrixXd> L_P(col_p, nbasis, nbasis);
        Tab_loc.noalias() = L_P * Cab_occ;
      }
      Ta_batch.middleCols(count_in_batch * na, na) = Tab_loc.leftCols(na);
      Tb_batch.middleCols(count_in_batch * nb, nb) = Tab_loc.rightCols(nb);
      count_in_batch++;

      if (count_in_batch == batch_p) {
        Ka_tls[tid].noalias() += Ta_batch * Ta_batch.transpose();
        Kb_tls[tid].noalias() += Tb_batch * Tb_batch.transpose();
        count_in_batch = 0;
      }
    }
    if (count_in_batch > 0) {
      auto Ta_rem = Ta_batch.leftCols(count_in_batch * na);
      auto Tb_rem = Tb_batch.leftCols(count_in_batch * nb);
      Ka_tls[tid].noalias() += Ta_rem * Ta_rem.transpose();
      Kb_tls[tid].noalias() += Tb_rem * Tb_rem.transpose();
    }
  }

  for (int t = 0; t < max_threads; ++t) {
    Ka_acc.noalias() += Ka_tls[t];
    Kb_acc.noalias() += Kb_tls[t];
  }
}"""

    src = replace_cpp_function(src, r"void\s+compute_exchange_cholesky_single\s*\([^)]*\)", new_single)
    src = replace_cpp_function(src, r"void\s+compute_exchange_cholesky_dual\s*\([^)]*\)", new_dual)
    return src, "PATCHED (fix auto-OOC reload bug + adaptive BATCH_P <=8MB/thread)"


# ==============================================================================
# PATCH 8: src/mp2/mp2.cc
# ==============================================================================

def patch_mp2_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_MP2_GUARDS]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    src = ensure_include(src, "<cstdlib>")
    src = ensure_include(src, "<stdexcept>")

    pat_df = r"df_eng\.compute\(false\);\s*scf_\.L_mat\s*=\s*std::move\(df_eng\.get_B_mat_mutable\(\)\);"
    rep_df = f"""{marker}
    double mem_lim_gb = std::max(0.5, config_.memory_limit_gb);
    if (const char* env_mem = std::getenv("MSHQC_MEMORY_GB")) {{
      const double v = std::atof(env_mem);
      if (v > mem_lim_gb) mem_lim_gb = v;
    }}
    const double est_df_gb =
        static_cast<double>(integrals::ao_tri_size(nbf_)) *
        static_cast<double>(aux_basis.n_basis_functions()) *
        static_cast<double>(sizeof(double)) / 1.0e9;
    if (est_df_gb > mem_lim_gb * 0.75) {{
      throw std::runtime_error(
          "DF-MP2/OMP2 requires " + std::to_string(est_df_gb) +
          " GB RAM for packed 3-center integrals, exceeding budget (" +
          std::to_string(mem_lim_gb * 0.75) +
          " GB). Increase memory_limit_gb / MSHQC_MEMORY_GB.");
    }}
    df_eng.compute(false);
    scf_.L_mat = std::move(df_eng.get_B_mat_mutable());"""
    src, n_df = sub_literal(pat_df, rep_df, src, count=1)
    if n_df != 1:
        raise RuntimeError("Anchor df_eng.compute(false) tidak ditemukan di src/mp2/mp2.cc")

    pat_bp = r"constexpr\s+int\s+BATCH_P\s*=\s*32;"
    rep_bp = (
        "const int max_o_u = std::max(1, nocc_a_ + nocc_b_);\n"
        "      const int BATCH_P = std::clamp(\n"
        "          static_cast<int>((8ULL * 1024ULL * 1024ULL) /\n"
        "                           std::max<size_t>(1, static_cast<size_t>(nbf_) * max_o_u * sizeof(double))),\n"
        "          1, 32);"
    )
    src, n_bp = sub_literal(pat_bp, rep_bp, src, count=1)
    if n_bp != 1:
        raise RuntimeError("Anchor BATCH_P = 32 tidak ditemukan di src/mp2/mp2.cc")

    return src, "PATCHED (BaseMP2 DF OOM guard + adaptive BATCH_P di UMP2 polish)"


# ==============================================================================
# PATCH 9: src/mp3/mp3.cc
# ==============================================================================

def patch_mp3_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_OMP3_HEAVY_RAM]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat_mv = (
        r"L2_aa_\s*=\s*t2_3rd_aa_;\s*"
        r"if\s*\(!is_restricted\s*&&\s*nb_\s*>\s*0\s*&&\s*vb_\s*>\s*0\)\s*\{\s*"
        r"L2_bb_\s*=\s*t2_3rd_bb_;\s*"
        r"L2_ab_\s*=\s*t2_3rd_ab_;\s*"
        r"\}"
    )
    rep_mv = (
        f"{marker}\n"
        "  L2_aa_ = std::move(t2_3rd_aa_);\n"
        "  if (!is_restricted && nb_ > 0 && vb_ > 0) {\n"
        "    L2_bb_ = std::move(t2_3rd_bb_);\n"
        "    L2_ab_ = std::move(t2_3rd_ab_);\n"
        "  }"
    )
    src, n_mv = sub_literal(pat_mv, rep_mv, src, count=1)
    if n_mv != 1:
        raise RuntimeError("Anchor L2_aa_ = t2_3rd_aa_ tidak ditemukan di src/mp3/mp3.cc")

    new_opdm_beta = """void OMP3::build_opdm_beta() {
  if (L2_bb_.size() == 0 && L2_ab_.size() == 0) {
    OMP2::build_opdm_beta();
    return;
  }
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
  G_oo_beta_ = Eigen::MatrixXd::Zero(nb_, nb_);
  G_vv_beta_ = Eigen::MatrixXd::Zero(vb_, vb_);
  if (is_restricted || nb_ <= 0 || vb_ <= 0) return;

  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  const int na_act = na_ - nf_a;
  const int nb_act = nb_ - nf_b;
  if (nb_act <= 0) return;

  auto* t2_bb_dense = t2_bb_.get_block(0, 0, 0, 0);
  auto* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);

  const int max_t = omp_get_max_threads();
  std::vector<Eigen::MatrixXd> Goo_tls(max_t, Eigen::MatrixXd::Zero(nb_act, nb_act));
  std::vector<Eigen::MatrixXd> Gvv_tls(max_t, Eigen::MatrixXd::Zero(vb_, vb_));

  if (t2_bb_dense && L2_bb_.size() > 0 && nb_act > 1) {
    const int a_tile = std::clamp(
        static_cast<int>((4ULL * 1024ULL * 1024ULL) /
                         std::max<size_t>(1, static_cast<size_t>(nb_act) * vb_ * sizeof(double))),
        1, vb_);
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      Eigen::MatrixXd& Goo_loc = Goo_tls[tid];
      Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];
      Eigen::MatrixXd T_bb(vb_, vb_), L_bb(vb_, vb_), Sum_bb(vb_, vb_);

      const int n_ij = nb_act * nb_act;
      #pragma omp for schedule(dynamic, 1) nowait
      for (int p = 0; p < n_ij; ++p) {
        const int i = nf_b + p / nb_act;
        const int j = nf_b + p % nb_act;
        if (i == j) continue;
        for (int b = 0; b < vb_; ++b) {
          for (int a = 0; a < vb_; ++a) {
            T_bb(a, b) = (*t2_bb_dense)(i, j, a, b);
            L_bb(a, b) = L2_bb_(i, j, a, b);
          }
        }
        Sum_bb.noalias() = T_bb + L_bb;
        Gvv_loc.noalias() += 0.5 * (T_bb * Sum_bb.transpose() + L_bb * T_bb.transpose());
      }

      Eigen::MatrixXd MatT(nb_act, a_tile * vb_);
      Eigen::MatrixXd MatL(nb_act, a_tile * vb_);
      Eigen::MatrixXd MatSum(nb_act, a_tile * vb_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int k = nf_b; k < nb_; ++k) {
        for (int a0 = 0; a0 < vb_; a0 += a_tile) {
          const int alen = std::min(a_tile, vb_ - a0);
          const int cols = alen * vb_;
          for (int da = 0; da < alen; ++da) {
            const int a = a0 + da;
            for (int b = 0; b < vb_; ++b) {
              const int col = da * vb_ + b;
              for (int ii = 0; ii < nb_act; ++ii) {
                const double tv = (*t2_bb_dense)(nf_b + ii, k, a, b);
                const double lv = L2_bb_(nf_b + ii, k, a, b);
                MatT(ii, col) = tv;
                MatL(ii, col) = lv;
                MatSum(ii, col) = tv + lv;
              }
            }
          }
          Goo_loc.noalias() -= 0.5 * (MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                                    + MatL.leftCols(cols) * MatT.leftCols(cols).transpose());
        }
      }
    }
  }

  if (t2_ab_dense && L2_ab_.size() > 0 && na_act > 0 && va_ > 0) {
    const int b_tile = std::clamp(
        static_cast<int>((4ULL * 1024ULL * 1024ULL) /
                         std::max<size_t>(1, static_cast<size_t>(nb_act) * va_ * sizeof(double))),
        1, vb_);
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      Eigen::MatrixXd& Goo_loc = Goo_tls[tid];
      Eigen::MatrixXd& Gvv_loc = Gvv_tls[tid];
      Eigen::MatrixXd T_ba(vb_, va_), L_ba(vb_, va_), Sum_ba(vb_, va_);

      const int n_ab = na_act * nb_act;
      #pragma omp for schedule(dynamic, 1) nowait
      for (int p = 0; p < n_ab; ++p) {
        const int i = nf_a + p / nb_act;
        const int j = nf_b + p % nb_act;
        for (int a = 0; a < va_; ++a) {
          for (int b = 0; b < vb_; ++b) {
            T_ba(b, a) = (*t2_ab_dense)(i, j, a, b);
            L_ba(b, a) = L2_ab_(i, j, a, b);
          }
        }
        Sum_ba.noalias() = T_ba + L_ba;
        Gvv_loc.noalias() += T_ba * Sum_ba.transpose() + L_ba * T_ba.transpose();
      }

      Eigen::MatrixXd MatT(nb_act, b_tile * va_);
      Eigen::MatrixXd MatL(nb_act, b_tile * va_);
      Eigen::MatrixXd MatSum(nb_act, b_tile * va_);
      #pragma omp for schedule(dynamic, 1) nowait
      for (int k = nf_a; k < na_; ++k) {
        for (int b0 = 0; b0 < vb_; b0 += b_tile) {
          const int blen = std::min(b_tile, vb_ - b0);
          const int cols = blen * va_;
          for (int db = 0; db < blen; ++db) {
            const int b = b0 + db;
            for (int a = 0; a < va_; ++a) {
              const int col = db * va_ + a;
              for (int jj = 0; jj < nb_act; ++jj) {
                const double tv = (*t2_ab_dense)(k, nf_b + jj, a, b);
                const double lv = L2_ab_(k, nf_b + jj, a, b);
                MatT(jj, col) = tv;
                MatL(jj, col) = lv;
                MatSum(jj, col) = tv + lv;
              }
            }
          }
          Goo_loc.noalias() -= MatT.leftCols(cols) * MatSum.leftCols(cols).transpose()
                             + MatL.leftCols(cols) * MatT.leftCols(cols).transpose();
        }
      }
    }
  }

  for (int t = 0; t < max_t; ++t) {
    G_oo_beta_.block(nf_b, nf_b, nb_act, nb_act).noalias() += Goo_tls[t];
    G_vv_beta_.noalias() += Gvv_tls[t];
  }
}"""

    new_build_gen_fock = """void OMP3::build_generalized_fock() {
  const bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

  const Eigen::MatrixXd& Cao = scf_.C_alpha.leftCols(na_);
  const Eigen::MatrixXd& Cav = scf_.C_alpha.rightCols(va_);
  Eigen::MatrixXd Z_mat_a = Eigen::MatrixXd::Zero(va_, na_);

  const Eigen::MatrixXd& Cbo = scf_.C_beta.leftCols(nb_);
  const Eigen::MatrixXd& Cbv = scf_.C_beta.rightCols(vb_);
  Eigen::MatrixXd Z_mat_b = Eigen::MatrixXd::Zero(vb_, nb_);

  auto* t_aa_dense = t2_aa_.get_block(0, 0, 0, 0);
  if (!t_aa_dense) return;

  const int n_aux = static_cast<int>(scf_.L_mat.cols());
  const bool is_ex = (config_.eri_method == "exact");
  const int nf_a = std::clamp(n_frozen_, 0, na_);
  const int nf_b = std::clamp(n_frozen_, 0, nb_);
  const int na_act = na_ - nf_a;
  const int nb_act = nb_ - nf_b;
  const double f_ladd = is_restricted ? 0.25 : 0.125;

  if (!is_ex) {
    auto make_bov_view = [&](const Eigen::MatrixXd& B_ov, int no, int nv) {
      return varray_view<double>(
          {(len_type)no, (len_type)nv, (len_type)n_aux},
          const_cast<double*>(B_ov.data()),
          {(stride_type)nv, 1, (stride_type)(no * nv)}
      );
    };
    auto make_bvv_view = [&](const Eigen::MatrixXd& B_vv, int nv) {
      return varray_view<double>(
          {(len_type)nv, (len_type)nv, (len_type)n_aux},
          const_cast<double*>(B_vv.data()),
          {(stride_type)nv, 1, (stride_type)(nv * nv)}
      );
    };
    auto slice_boo_act = [&](const Eigen::MatrixXd& Boo, int n_full, int nf) -> Eigen::MatrixXd {
      const int n_act = n_full - nf;
      if (nf == 0) return Boo;
      Eigen::MatrixXd out(static_cast<Eigen::Index>(n_act) * n_act, n_aux);
      #pragma omp parallel for schedule(static)
      for (int P = 0; P < n_aux; ++P) {
        const double* s = Boo.col(P).data();
        double* d = out.col(P).data();
        for (int j = 0; j < n_act; ++j)
          for (int i = 0; i < n_act; ++i)
            d[i + j * n_act] = s[(nf + i) + (nf + j) * n_full];
      }
      return out;
    };

    auto accum_vvvv_ladder_act = [&](
        double scale,
        const Eigen::Tensor<double, 4>& T_bra, int no1, int no2, int nv1, int nv2,
        const Eigen::Tensor<double, 4>& T_ket,
        const Eigen::MatrixXd& B_ov_m, int nm, int nv_m,
        const Eigen::MatrixXd& B_vv_r, int nv_r,
        bool swap_ab_mode,
        Eigen::MatrixXd& Z_target_act)
    {
      if (nm <= 0 || no1 <= 0 || no2 <= 0 || nv1 <= 0 || nv2 <= 0 || n_aux <= 0) return;
      constexpr size_t kMaxChunkBytes = 64ULL * 1024ULL * 1024ULL;
      const size_t slice_bytes = static_cast<size_t>(nv_m) * nv_r * nv_r * sizeof(double);
      const int mb_max = (slice_bytes > 0)
          ? std::max(1, static_cast<int>(kMaxChunkBytes / slice_bytes))
          : nm;

      auto& T_bra_nc = const_cast<Eigen::Tensor<double, 4>&>(T_bra);
      auto& T_ket_nc = const_cast<Eigen::Tensor<double, 4>&>(T_ket);
      TBLIS_VIEW_4D(t_Tbra, T_bra_nc, no1, no2, nv1, nv2);
      TBLIS_VIEW_4D(t_Tket, T_ket_nc, no1, no2, nv1, nv2);

      for (int m0 = 0; m0 < nm; m0 += mb_max) {
        const int mb = std::min(mb_max, nm - m0);
        Eigen::MatrixXd B_chunk_buf;
        const Eigen::MatrixXd* B_chunk_ptr = &B_ov_m;
        if (m0 != 0 || mb != nm) {
          B_chunk_buf = B_ov_m.middleRows(static_cast<Eigen::Index>(m0) * nv_m,
                                          static_cast<Eigen::Index>(mb) * nv_m);
          B_chunk_ptr = &B_chunk_buf;
        }
        Eigen::MatrixXd M_chunk = parallel_gemm_abt_mp3(*B_chunk_ptr, B_vv_r);
        auto t_Vchunk = make_mat_view_4d_mp3(M_chunk, mb, nv_m, nv_r, nv_r);

        Eigen::Tensor<double, 4> Y_chunk(no1, no2, mb, nv_r);
        Y_chunk.setZero();
        TBLIS_VIEW_4D(t_Ychunk, Y_chunk, no1, no2, mb, nv_r);
        TBLIS_VIEW_2D(t_Zslice, Z_target_act.col(m0).data(), nv_m, mb);

        if (!swap_ab_mode) {
          tblis::mult<double>(1.0, t_Tbra, "ijab", t_Vchunk, "mafb", 0.0, t_Ychunk, "ijmf");
          tblis::mult<double>(scale, t_Tket, "ijxf", t_Ychunk, "ijmf", 1.0, t_Zslice, "xm");
        } else {
          tblis::mult<double>(1.0, t_Tbra, "ijab", t_Vchunk, "mbea", 0.0, t_Ychunk, "ijme");
          tblis::mult<double>(scale, t_Tket, "ijex", t_Ychunk, "ijme", 1.0, t_Zslice, "xm");
        }
      }
    };

    if (na_act > 0 && va_ > 0) {
      Eigen::MatrixXd Z_act_a = Eigen::MatrixXd::Zero(va_, na_act);
      TBLIS_VIEW_2D(t_Za, Z_act_a.data(), va_, na_act);

      Eigen::MatrixXd B_ij_a = slice_boo_act(B_oo_P_alpha_, na_, nf_a);
      Eigen::MatrixXd B_ia_a = (nf_a > 0)
          ? B_ia_P_alpha_.bottomRows(static_cast<Eigen::Index>(na_act) * va_).eval()
          : B_ia_P_alpha_;
      const Eigen::MatrixXd& B_ab_a = B_vv_P_alpha_;

      auto t_Bov_a = make_bov_view(B_ia_a, na_act, va_);
      auto t_Bvv_a = make_bvv_view(B_ab_a, va_);

      Eigen::MatrixXd M_ooov_aa = parallel_gemm_abt_mp3(B_ij_a, B_ia_a);
      auto t_Vooov_aa = make_mat_view_4d_mp3(M_ooov_aa, na_act, na_act, na_act, va_);

      {
        Eigen::Tensor<double, 4> Teff_aa(na_act, na_act, va_, va_);
        #pragma omp parallel for collapse(4) schedule(static)
        for (int i = 0; i < na_act; ++i)
          for (int j = 0; j < na_act; ++j)
            for (int a = 0; a < va_; ++a)
              for (int b = 0; b < va_; ++b) {
                const double t_val = (*t_aa_dense)(nf_a + i, a, nf_a + j, b);
                if (is_restricted) {
                  const double t_t = 2.0 * t_val - (*t_aa_dense)(nf_a + i, b, nf_a + j, a);
                  const double l_t = 2.0 * L2_aa_(nf_a + i, nf_a + j, a, b) - L2_aa_(nf_a + i, nf_a + j, b, a);
                  Teff_aa(i, j, a, b) = t_t + l_t;
                } else {
                  Teff_aa(i, j, a, b) = t_val + L2_aa_(nf_a + i, nf_a + j, a, b);
                }
              }
        TBLIS_VIEW_4D(t_Teff_aa, Teff_aa, na_act, na_act, va_, va_);
        Eigen::Tensor<double, 3> X_teff_a(na_act, va_, n_aux);
        X_teff_a.setZero();
        TBLIS_VIEW_3D(t_Xteff_a, X_teff_a.data(), na_act, va_, n_aux);
        tblis::mult<double>(1.0, t_Teff_aa, "ikbc", t_Bov_a, "kcP", 0.0, t_Xteff_a, "ibP");
        tblis::mult<double>(1.0, t_Bvv_a, "abP", t_Xteff_a, "ibP", 0.0, t_Za, "ai");
        tblis::mult<double>(-1.0, t_Vooov_aa, "jikc", t_Teff_aa, "jkac", 1.0, t_Za, "ai");
      }

      Eigen::Tensor<double, 4> T2_aa(na_act, na_act, va_, va_);
      Eigen::Tensor<double, 4> T2_tilde_aa;
      if (is_restricted) T2_tilde_aa.resize(na_act, na_act, va_, va_);
      #pragma omp parallel for collapse(4) schedule(static)
      for (int i = 0; i < na_act; ++i)
        for (int j = 0; j < na_act; ++j)
          for (int a = 0; a < va_; ++a)
            for (int b = 0; b < va_; ++b) {
              const double tv = (*t_aa_dense)(nf_a + i, a, nf_a + j, b);
              T2_aa(i, j, a, b) = tv;
              if (is_restricted) {
                T2_tilde_aa(i, j, a, b) = 2.0 * tv - (*t_aa_dense)(nf_a + i, b, nf_a + j, a);
              }
            }
      Eigen::Tensor<double, 4>& T2t_ref = is_restricted ? T2_tilde_aa : T2_aa;
      TBLIS_VIEW_4D(t_T2, T2_aa, na_act, na_act, va_, va_);
      TBLIS_VIEW_4D(t_T2t, T2t_ref, na_act, na_act, va_, va_);

      const bool has_b_act = (!is_restricted && nb_act > 0 && vb_ > 0);
      Eigen::Tensor<double, 4> T2_bb_act, T2_ab_act;
      if (has_b_act) {
        auto* t2_bb_d = t2_bb_.get_block(0, 0, 0, 0);
        auto* t2_ab_d = t2_ab_.get_block(0, 0, 0, 0);
        T2_bb_act = Eigen::Tensor<double, 4>(nb_act, nb_act, vb_, vb_); T2_bb_act.setZero();
        T2_ab_act = Eigen::Tensor<double, 4>(na_act, nb_act, va_, vb_); T2_ab_act.setZero();
        if (t2_bb_d) {
          #pragma omp parallel for collapse(4) schedule(static)
          for (int i = 0; i < nb_act; ++i)
            for (int j = 0; j < nb_act; ++j)
              for (int a = 0; a < vb_; ++a)
                for (int b = 0; b < vb_; ++b)
                  T2_bb_act(i, j, a, b) = (*t2_bb_d)(nf_b + i, nf_b + j, a, b);
        }
        if (t2_ab_d) {
          #pragma omp parallel for collapse(4) schedule(static)
          for (int i = 0; i < na_act; ++i)
            for (int j = 0; j < nb_act; ++j)
              for (int a = 0; a < va_; ++a)
                for (int b = 0; b < vb_; ++b)
                  T2_ab_act(i, j, a, b) = (*t2_ab_d)(nf_a + i, nf_b + j, a, b);
        }
      }

      accum_vvvv_ladder_act(
          -4.0 * f_ladd,
          T2t_ref, na_act, na_act, va_, va_, T2_aa,
          B_ia_a, na_act, va_, B_ab_a, va_,
          false, Z_act_a);

      {
        Eigen::Tensor<double, 4> G_oooo_aa(na_act, na_act, na_act, na_act);
        G_oooo_aa.setZero();
        TBLIS_VIEW_4D(t_Goooo_aa, G_oooo_aa, na_act, na_act, na_act, na_act);
        tblis::mult<double>(f_ladd, t_T2t, "ijab", t_T2, "mnab", 0.0, t_Goooo_aa, "minj");
        tblis::mult<double>(4.0, t_Goooo_aa, "minj", t_Vooov_aa, "njix", 1.0, t_Za, "xm");
      }

      {
        Eigen::Tensor<double, 4> G_ovov_aa(na_act, va_, na_act, va_);
        G_ovov_aa.setZero();
        TBLIS_VIEW_4D(t_Govov_aa, G_ovov_aa, na_act, va_, na_act, va_);
        if (is_restricted) {
          tblis::mult<double>( 0.50, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
          tblis::mult<double>(-0.25, t_T2, "kjbc", t_T2t, "ijab", 1.0, t_Govov_aa, "iakc");
          tblis::mult<double>( 0.50, t_T2, "miea", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
          tblis::mult<double>(-0.25, t_T2, "miae", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
        } else {
          tblis::mult<double>(0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
          if (has_b_act) {
            TBLIS_VIEW_4D(t_Tab_tmp, T2_ab_act, na_act, nb_act, va_, vb_);
            tblis::mult<double>(0.5, t_Tab_tmp, "kjcb", t_Tab_tmp, "ijab", 1.0, t_Govov_aa, "iakc");
          }
        }
        Eigen::Tensor<double, 3> W_ov_a(na_act, va_, n_aux);
        W_ov_a.setZero();
        TBLIS_VIEW_3D(t_Wov_a, W_ov_a.data(), na_act, va_, n_aux);
        if (is_restricted) {
          tblis::mult<double>(1.0, t_Govov_aa, "makc", t_Bov_a, "kcP", 0.0, t_Wov_a, "maP");
          tblis::mult<double>(2.0, t_Bvv_a, "xaP", t_Wov_a, "maP", 1.0, t_Za, "xm");
          tblis::mult<double>(-2.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Za, "xm");
        } else {
          tblis::mult<double>(1.0, t_Govov_aa, "makc", t_Bov_a, "kcP", 0.0, t_Wov_a, "maP");
          tblis::mult<double>(1.0, t_Govov_aa, "iamc", t_Bov_a, "iaP", 1.0, t_Wov_a, "mcP");
          tblis::mult<double>(1.0, t_Bvv_a, "xaP", t_Wov_a, "maP", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Govov_aa, "iakx", t_Vooov_aa, "kmia", 1.0, t_Za, "xm");
        }
      }

      {
        Eigen::Tensor<double, 4> G_oovv_aa(na_act, na_act, va_, va_);
        G_oovv_aa.setZero();
        TBLIS_VIEW_4D(t_Goovv_aa, G_oovv_aa, na_act, na_act, va_, va_);
        if (is_restricted) {
          tblis::mult<double>(-0.25, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
          tblis::mult<double>(-0.25, t_T2, "miea", t_T2t, "mkec", 1.0, t_Goovv_aa, "ikac");
          tblis::mult<double>(-0.50, t_T2, "kmec", t_T2t, "imea", 1.0, t_Goovv_aa, "ikac");
        } else {
          tblis::mult<double>(-0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
          if (has_b_act) {
            TBLIS_VIEW_4D(t_Tab_tmp, T2_ab_act, na_act, nb_act, va_, vb_);
            tblis::mult<double>(-0.5, t_Tab_tmp, "kjcb", t_Tab_tmp, "ijab", 1.0, t_Goovv_aa, "ikac");
          }
        }
        Eigen::Tensor<double, 3> U_oo_a(na_act, na_act, n_aux);
        U_oo_a.setZero();
        TBLIS_VIEW_3D(t_Uoo_a, U_oo_a.data(), na_act, na_act, n_aux);
        tblis::mult<double>(1.0, t_Goovv_aa, "mkac", t_Bvv_a, "caP", 0.0, t_Uoo_a, "mkP");
        tblis::mult<double>(1.0, t_Bov_a, "kxP", t_Uoo_a, "mkP", 1.0, t_Za, "xm");
        tblis::mult<double>(1.0, t_Goovv_aa, "imac", t_Bvv_a, "acP", 0.0, t_Uoo_a, "imP");
        tblis::mult<double>(1.0, t_Bov_a, "ixP", t_Uoo_a, "imP", 1.0, t_Za, "xm");
        tblis::mult<double>(-1.0, t_Goovv_aa, "ikxc", t_Vooov_aa, "ikmc", 1.0, t_Za, "xm");
        tblis::mult<double>(-1.0, t_Goovv_aa, "ikax", t_Vooov_aa, "ikma", 1.0, t_Za, "xm");
      }

      M_ooov_aa.resize(0, 0);
      T2_tilde_aa.resize(0, 0, 0, 0);

      if (has_b_act) {
        Eigen::MatrixXd Z_act_b = Eigen::MatrixXd::Zero(vb_, nb_act);
        TBLIS_VIEW_2D(t_Zb, Z_act_b.data(), vb_, nb_act);
        TBLIS_VIEW_4D(t_T2_bb, T2_bb_act, nb_act, nb_act, vb_, vb_);
        TBLIS_VIEW_4D(t_T2_ab, T2_ab_act, na_act, nb_act, va_, vb_);

        Eigen::MatrixXd B_ij_b = slice_boo_act(B_oo_P_beta_, nb_, nf_b);
        Eigen::MatrixXd B_ia_b = (nf_b > 0)
            ? B_ia_P_beta_.bottomRows(static_cast<Eigen::Index>(nb_act) * vb_).eval()
            : B_ia_P_beta_;
        const Eigen::MatrixXd& B_ab_b = B_vv_P_beta_;

        auto t_Bov_b = make_bov_view(B_ia_b, nb_act, vb_);
        auto t_Bvv_b = make_bvv_view(B_ab_b, vb_);

        Eigen::MatrixXd M_ooov_bb = parallel_gemm_abt_mp3(B_ij_b, B_ia_b);
        Eigen::MatrixXd M_ooov_aa_bb = parallel_gemm_abt_mp3(B_ij_a, B_ia_b);
        Eigen::MatrixXd M_ooov_bb_aa = parallel_gemm_abt_mp3(B_ij_b, B_ia_a);

        auto t_Vooov_bb = make_mat_view_4d_mp3(M_ooov_bb, nb_act, nb_act, nb_act, vb_);
        auto t_Vooov_aa_bb = make_mat_view_4d_mp3(M_ooov_aa_bb, na_act, na_act, nb_act, vb_);
        auto t_Vooov_bb_aa = make_mat_view_4d_mp3(M_ooov_bb_aa, nb_act, nb_act, na_act, va_);

        {
          Eigen::Tensor<double, 3> X_b(nb_act, vb_, n_aux);
          X_b.setZero();
          TBLIS_VIEW_3D(t_Xb, X_b.data(), nb_act, vb_, n_aux);

          {
            Eigen::Tensor<double, 4> Teff_bb(nb_act, nb_act, vb_, vb_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < nb_act; ++i)
              for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < vb_; ++a)
                  for (int b = 0; b < vb_; ++b)
                    Teff_bb(i, j, a, b) = T2_bb_act(i, j, a, b) + L2_bb_(nf_b + i, nf_b + j, a, b);
            TBLIS_VIEW_4D(t_Teff_bb, Teff_bb, nb_act, nb_act, vb_, vb_);
            tblis::mult<double>(1.0, t_Teff_bb, "ikbc", t_Bov_b, "kcP", 0.0, t_Xb, "ibP");
            tblis::mult<double>(-1.0, t_Vooov_bb, "jikc", t_Teff_bb, "jkac", 1.0, t_Zb, "ai");
          }

          {
            Eigen::Tensor<double, 4> Teff_ab(na_act, nb_act, va_, vb_);
            #pragma omp parallel for collapse(4) schedule(static)
            for (int i = 0; i < na_act; ++i)
              for (int j = 0; j < nb_act; ++j)
                for (int a = 0; a < va_; ++a)
                  for (int b = 0; b < vb_; ++b)
                    Teff_ab(i, j, a, b) = T2_ab_act(i, j, a, b) + L2_ab_(nf_a + i, nf_b + j, a, b);
            TBLIS_VIEW_4D(t_Teff_ab, Teff_ab, na_act, nb_act, va_, vb_);

            Eigen::Tensor<double, 3> X_ab_a(na_act, va_, n_aux);
            X_ab_a.setZero();
            TBLIS_VIEW_3D(t_Xab_a, X_ab_a.data(), na_act, va_, n_aux);
            tblis::mult<double>(1.0, t_Teff_ab, "ikbc", t_Bov_b, "kcP", 0.0, t_Xab_a, "ibP");
            tblis::mult<double>(1.0, t_Bvv_a, "abP", t_Xab_a, "ibP", 1.0, t_Za, "ai");
            tblis::mult<double>(-1.0, t_Vooov_aa_bb, "jikc", t_Teff_ab, "jkac", 1.0, t_Za, "ai");

            tblis::mult<double>(1.0, t_Teff_ab, "kicb", t_Bov_a, "kcP", 1.0, t_Xb, "ibP");
            tblis::mult<double>(-1.0, t_Vooov_bb_aa, "jikc", t_Teff_ab, "kjca", 1.0, t_Zb, "ai");
          }

          tblis::mult<double>(1.0, t_Bvv_b, "abP", t_Xb, "ibP", 1.0, t_Zb, "ai");
        }

        accum_vvvv_ladder_act(-0.5, T2_bb_act, nb_act, nb_act, vb_, vb_, T2_bb_act,
                              B_ia_b, nb_act, vb_, B_ab_b, vb_, false, Z_act_b);
        accum_vvvv_ladder_act(-1.0, T2_ab_act, na_act, nb_act, va_, vb_, T2_ab_act,
                              B_ia_a, na_act, va_, B_ab_b, vb_, false, Z_act_a);
        accum_vvvv_ladder_act(-1.0, T2_ab_act, na_act, nb_act, va_, vb_, T2_ab_act,
                              B_ia_b, nb_act, vb_, B_ab_a, va_, true, Z_act_b);

        {
          Eigen::Tensor<double, 4> G_oooo_bb(nb_act, nb_act, nb_act, nb_act); G_oooo_bb.setZero();
          Eigen::Tensor<double, 4> G_oooo_ab(na_act, na_act, nb_act, nb_act); G_oooo_ab.setZero();
          TBLIS_VIEW_4D(t_Goooo_bb, G_oooo_bb, nb_act, nb_act, nb_act, nb_act);
          TBLIS_VIEW_4D(t_Goooo_ab, G_oooo_ab, na_act, na_act, nb_act, nb_act);
          tblis::mult<double>(0.125, t_T2_bb, "ijab", t_T2_bb, "mnab", 0.0, t_Goooo_bb, "minj");
          tblis::mult<double>(0.500, t_T2_ab, "ijab", t_T2_ab, "mnab", 0.0, t_Goooo_ab, "minj");

          tblis::mult<double>(1.0, t_Goooo_ab, "minj", t_Vooov_bb_aa, "njix", 1.0, t_Za, "xm");
          tblis::mult<double>(1.0, t_Goooo_ab, "imnj", t_Vooov_bb_aa, "njix", 1.0, t_Za, "xm");
          tblis::mult<double>(1.0, t_Goooo_ab, "inmj", t_Vooov_aa_bb, "injx", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_ab, "injm", t_Vooov_aa_bb, "injx", 1.0, t_Zb, "xm");

          tblis::mult<double>(1.0, t_Goooo_bb, "minj", t_Vooov_bb, "njix", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_bb, "imnj", t_Vooov_bb, "njix", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_bb, "inmj", t_Vooov_bb, "injx", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goooo_bb, "injm", t_Vooov_bb, "injx", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_ovov_bb(nb_act, vb_, nb_act, vb_); G_ovov_bb.setZero();
          TBLIS_VIEW_4D(t_Govov_bb, G_ovov_bb, nb_act, vb_, nb_act, vb_);
          tblis::mult<double>(0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Govov_bb, "iakc");
          tblis::mult<double>(0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Govov_bb, "iakc");

          Eigen::Tensor<double, 3> W_ov_b(nb_act, vb_, n_aux); W_ov_b.setZero();
          TBLIS_VIEW_3D(t_Wov_b, W_ov_b.data(), nb_act, vb_, n_aux);
          tblis::mult<double>(1.0, t_Govov_bb, "makc", t_Bov_b, "kcP", 0.0, t_Wov_b, "maP");
          tblis::mult<double>(1.0, t_Govov_bb, "iamc", t_Bov_b, "iaP", 1.0, t_Wov_b, "mcP");
          tblis::mult<double>(1.0, t_Bvv_b, "xaP", t_Wov_b, "maP", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Govov_bb, "ixkc", t_Vooov_bb, "imkc", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Govov_bb, "iakx", t_Vooov_bb, "kmia", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_ovov_ab(na_act, va_, nb_act, vb_); G_ovov_ab.setZero();
          TBLIS_VIEW_4D(t_Govov_ab, G_ovov_ab, na_act, va_, nb_act, vb_);
          tblis::mult<double>(1.0, t_T2, "miea", t_T2_ab, "mkec", 0.0, t_Govov_ab, "iakc");
          tblis::mult<double>(1.0, t_T2_bb, "kjcb", t_T2_ab, "ijab", 1.0, t_Govov_ab, "iakc");

          Eigen::Tensor<double, 3> W_ov_b(nb_act, vb_, n_aux); W_ov_b.setZero();
          TBLIS_VIEW_3D(t_Wov_b, W_ov_b.data(), nb_act, vb_, n_aux);
          tblis::mult<double>(1.0, t_Govov_ab, "iamc", t_Bov_a, "iaP", 0.0, t_Wov_b, "mcP");
          tblis::mult<double>(1.0, t_Bvv_b, "xaP", t_Wov_b, "maP", 1.0, t_Zb, "xm");

          Eigen::Tensor<double, 3> W_ab_a(na_act, va_, n_aux); W_ab_a.setZero();
          TBLIS_VIEW_3D(t_Wab_a, W_ab_a.data(), na_act, va_, n_aux);
          tblis::mult<double>(1.0, t_Govov_ab, "makc", t_Bov_b, "kcP", 0.0, t_Wab_a, "maP");
          tblis::mult<double>(1.0, t_Bvv_a, "xaP", t_Wab_a, "maP", 1.0, t_Za, "xm");

          tblis::mult<double>(-1.0, t_Govov_ab, "ixkc", t_Vooov_aa_bb, "imkc", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Govov_ab, "iakx", t_Vooov_bb_aa, "kmia", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_oovv_bb(nb_act, nb_act, vb_, vb_); G_oovv_bb.setZero();
          TBLIS_VIEW_4D(t_Goovv_bb, G_oovv_bb, nb_act, nb_act, vb_, vb_);
          tblis::mult<double>(-0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Goovv_bb, "ikac");
          tblis::mult<double>(-0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Goovv_bb, "ikac");

          Eigen::Tensor<double, 3> U_oo_b(nb_act, nb_act, n_aux); U_oo_b.setZero();
          TBLIS_VIEW_3D(t_Uoo_b, U_oo_b.data(), nb_act, nb_act, n_aux);
          tblis::mult<double>(1.0, t_Goovv_bb, "mkac", t_Bvv_b, "caP", 0.0, t_Uoo_b, "mkP");
          tblis::mult<double>(1.0, t_Bov_b, "kxP", t_Uoo_b, "mkP", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Goovv_bb, "imac", t_Bvv_b, "acP", 0.0, t_Uoo_b, "imP");
          tblis::mult<double>(1.0, t_Bov_b, "ixP", t_Uoo_b, "imP", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_bb, "ikxc", t_Vooov_bb, "ikmc", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_bb, "ikax", t_Vooov_bb, "ikma", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_oovv_ab_ex(na_act, na_act, vb_, vb_); G_oovv_ab_ex.setZero();
          TBLIS_VIEW_4D(t_Goovv_ab_ex, G_oovv_ab_ex, na_act, na_act, vb_, vb_);
          tblis::mult<double>(-0.5, t_T2_ab, "kmec", t_T2_ab, "imea", 0.0, t_Goovv_ab_ex, "ikac");

          Eigen::Tensor<double, 3> U_ab_ex(na_act, na_act, n_aux); U_ab_ex.setZero();
          TBLIS_VIEW_3D(t_Uab_ex, U_ab_ex.data(), na_act, na_act, n_aux);
          tblis::mult<double>(1.0, t_Goovv_ab_ex, "mkac", t_Bvv_b, "acP", 0.0, t_Uab_ex, "mkP");
          tblis::mult<double>(1.0, t_Bov_a, "kxP", t_Uab_ex, "mkP", 1.0, t_Za, "xm");
          tblis::mult<double>(1.0, t_Bov_a, "ixP", t_Uab_ex, "imP", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikxc", t_Vooov_aa_bb, "ikmc", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikax", t_Vooov_aa_bb, "ikma", 1.0, t_Zb, "xm");
        }

        {
          Eigen::Tensor<double, 4> G_oovv_ba_ex(nb_act, nb_act, va_, va_); G_oovv_ba_ex.setZero();
          TBLIS_VIEW_4D(t_Goovv_ba_ex, G_oovv_ba_ex, nb_act, nb_act, va_, va_);
          tblis::mult<double>(-0.5, t_T2_ab, "mkce", t_T2_ab, "miae", 0.0, t_Goovv_ba_ex, "ikac");

          Eigen::Tensor<double, 3> U_ba_ex(nb_act, nb_act, n_aux); U_ba_ex.setZero();
          TBLIS_VIEW_3D(t_Uba_ex, U_ba_ex.data(), nb_act, nb_act, n_aux);
          tblis::mult<double>(1.0, t_Goovv_ba_ex, "mkac", t_Bvv_a, "acP", 0.0, t_Uba_ex, "mkP");
          tblis::mult<double>(1.0, t_Bov_b, "kxP", t_Uba_ex, "mkP", 1.0, t_Zb, "xm");
          tblis::mult<double>(1.0, t_Bov_b, "ixP", t_Uba_ex, "imP", 1.0, t_Zb, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikxc", t_Vooov_bb_aa, "ikmc", 1.0, t_Za, "xm");
          tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikax", t_Vooov_bb_aa, "ikma", 1.0, t_Za, "xm");
        }

        Z_mat_b.rightCols(nb_act) = Z_act_b;
      }

      Z_mat_a.rightCols(na_act) = Z_act_a;
      if (is_restricted) Z_mat_b = Z_mat_a;
    }
  } else {
    Eigen::Tensor<double, 4> Teff_aa(na_, na_, va_, va_);
    Eigen::Tensor<double, 4> T2_aa(na_, na_, va_, va_);
    Eigen::Tensor<double, 4> T2_tilde_aa(na_, na_, va_, va_);
    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < na_; ++i)
      for (int j = 0; j < na_; ++j)
        for (int a = 0; a < va_; ++a)
          for (int b = 0; b < va_; ++b) {
            const double t_val = (*t_aa_dense)(i, a, j, b);
            T2_aa(i, j, a, b) = t_val;
            if (is_restricted) {
              const double t_tilde = 2.0 * t_val - (*t_aa_dense)(i, b, j, a);
              const double l_tilde = 2.0 * L2_aa_(i, j, a, b) - L2_aa_(i, j, b, a);
              Teff_aa(i, j, a, b) = t_tilde + l_tilde;
              T2_tilde_aa(i, j, a, b) = t_tilde;
            } else {
              Teff_aa(i, j, a, b) = t_val + L2_aa_(i, j, a, b);
              T2_tilde_aa(i, j, a, b) = t_val;
            }
          }
    TBLIS_VIEW_4D(t_Teff_aa, Teff_aa, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T2, T2_aa, na_, na_, va_, va_);
    TBLIS_VIEW_4D(t_T2t, T2_tilde_aa, na_, na_, va_, va_);
    TBLIS_VIEW_2D(t_Zmat_a, Z_mat_a.data(), va_, na_);

    Eigen::Tensor<double, 4> G_ovov_aa(na_, va_, na_, va_); G_ovov_aa.setZero();
    Eigen::Tensor<double, 4> G_oovv_aa(na_, na_, va_, va_); G_oovv_aa.setZero();
    TBLIS_VIEW_4D(t_Govov_aa, G_ovov_aa, na_, va_, na_, va_);
    TBLIS_VIEW_4D(t_Goovv_aa, G_oovv_aa, na_, na_, va_, va_);

    if (is_restricted) {
      tblis::mult<double>( 0.50, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.25, t_T2, "kjbc", t_T2t, "ijab", 1.0, t_Govov_aa, "iakc");
      tblis::mult<double>( 0.50, t_T2, "miea", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.25, t_T2, "miae", t_T2t, "mkec", 1.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.25, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
      tblis::mult<double>(-0.25, t_T2, "miea", t_T2t, "mkec", 1.0, t_Goovv_aa, "ikac");
      tblis::mult<double>(-0.50, t_T2, "kmec", t_T2t, "imea", 1.0, t_Goovv_aa, "ikac");
    } else {
      tblis::mult<double>( 0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Govov_aa, "iakc");
      tblis::mult<double>(-0.5, t_T2, "kjcb", t_T2t, "ijab", 0.0, t_Goovv_aa, "ikac");
      if (nb_ > 0 && vb_ > 0) {
        auto* t2_ab_dense_tmp = t2_ab_.get_block(0, 0, 0, 0);
        if (t2_ab_dense_tmp) {
          TBLIS_VIEW_4D(t_T2_ab_tmp, (*t2_ab_dense_tmp), na_, nb_, va_, vb_);
          tblis::mult<double>( 0.5, t_T2_ab_tmp, "kjcb", t_T2_ab_tmp, "ijab", 1.0, t_Govov_aa, "iakc");
          tblis::mult<double>(-0.5, t_T2_ab_tmp, "kjcb", t_T2_ab_tmp, "ijab", 1.0, t_Goovv_aa, "ikac");
        }
      }
    }

    const auto& eri_ao_ref = integrals_->compute_eri();
    Eigen::Tensor<double, 4> half_ov_a = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cao, Cav, nbf_, na_, va_);
    Eigen::Tensor<double, 4> half_oo_a = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cao, Cao, nbf_, na_, na_);
    Eigen::Tensor<double, 4> V_ovvv_aa = integrals::ERITransformer::transform_finish_ket(half_ov_a, Cav, Cav, nbf_, na_, va_, va_, va_);
    Eigen::Tensor<double, 4> V_ooov_aa = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cao, Cav, nbf_, na_, na_, na_, va_);
    if (is_restricted || nb_ == 0 || vb_ == 0) {
      half_ov_a.resize(0, 0, 0, 0);
      half_oo_a.resize(0, 0, 0, 0);
    }

    TBLIS_VIEW_4D(t_Vovvv_aa, V_ovvv_aa, na_, va_, va_, va_);
    TBLIS_VIEW_4D(t_Vooov_aa, V_ooov_aa, na_, na_, na_, va_);

    tblis::mult<double>( 1.0, t_Vovvv_aa, "kcab", t_Teff_aa, "ikbc", 0.0, t_Zmat_a, "ai");
    tblis::mult<double>(-1.0, t_Vooov_aa, "jikc", t_Teff_aa, "jkac", 1.0, t_Zmat_a, "ai");

    Eigen::Tensor<double, 4> Y_ooov_aa(na_, na_, na_, va_); Y_ooov_aa.setZero();
    TBLIS_VIEW_4D(t_Yooov_aa, Y_ooov_aa, na_, na_, na_, va_);
    tblis::mult<double>(1.0, t_T2t, "ijab", t_Vovvv_aa, "mafb", 0.0, t_Yooov_aa, "ijmf");
    tblis::mult<double>(-4.0 * f_ladd, t_T2, "ijxf", t_Yooov_aa, "ijmf", 1.0, t_Zmat_a, "xm");

    Eigen::Tensor<double, 4> G_oooo_aa(na_, na_, na_, na_); G_oooo_aa.setZero();
    TBLIS_VIEW_4D(t_Goooo_aa, G_oooo_aa, na_, na_, na_, na_);
    tblis::mult<double>(f_ladd, t_T2t, "ijab", t_T2, "mnab", 0.0, t_Goooo_aa, "minj");
    tblis::mult<double>(4.0, t_Goooo_aa, "minj", t_Vooov_aa, "njix", 1.0, t_Zmat_a, "xm");

    if (is_restricted) {
      tblis::mult<double>( 2.0, t_Govov_aa, "makc", t_Vovvv_aa, "kcxa", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-2.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Zmat_a, "xm");
    } else {
      tblis::mult<double>( 1.0, t_Govov_aa, "makc", t_Vovvv_aa, "kcxa", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Govov_aa, "ixkc", t_Vooov_aa, "imkc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>( 1.0, t_Govov_aa, "iamc", t_Vovvv_aa, "iaxc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Govov_aa, "iakx", t_Vooov_aa, "kmia", 1.0, t_Zmat_a, "xm");
    }

    tblis::mult<double>( 1.0, t_Goovv_aa, "mkac", t_Vovvv_aa, "kxca", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>( 1.0, t_Goovv_aa, "imac", t_Vovvv_aa, "ixac", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(-1.0, t_Goovv_aa, "ikxc", t_Vooov_aa, "ikmc", 1.0, t_Zmat_a, "xm");
    tblis::mult<double>(-1.0, t_Goovv_aa, "ikax", t_Vooov_aa, "ikma", 1.0, t_Zmat_a, "xm");

    Teff_aa.resize(0, 0, 0, 0);
    T2_tilde_aa.resize(0, 0, 0, 0);
    G_ovov_aa.resize(0, 0, 0, 0);
    G_oovv_aa.resize(0, 0, 0, 0);

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
      auto* t2_bb_dense = t2_bb_.get_block(0, 0, 0, 0);
      Eigen::Tensor<double, 4> dummy_bb;
      if (!t2_bb_dense) { dummy_bb = Eigen::Tensor<double, 4>(nb_, nb_, vb_, vb_); dummy_bb.setZero(); t2_bb_dense = &dummy_bb; }
      auto* t2_ab_dense = t2_ab_.get_block(0, 0, 0, 0);
      Eigen::Tensor<double, 4> dummy_ab;
      if (!t2_ab_dense) { dummy_ab = Eigen::Tensor<double, 4>(na_, nb_, va_, vb_); dummy_ab.setZero(); t2_ab_dense = &dummy_ab; }

      TBLIS_VIEW_2D(t_Zmat_b, Z_mat_b.data(), vb_, nb_);
      TBLIS_VIEW_4D(t_T2_bb, (*t2_bb_dense), nb_, nb_, vb_, vb_);
      TBLIS_VIEW_4D(t_T2_ab, (*t2_ab_dense), na_, nb_, va_, vb_);

      Eigen::Tensor<double, 4> Teff_bb(nb_, nb_, vb_, vb_);
      Eigen::Tensor<double, 4> Teff_ab(na_, nb_, va_, vb_);
      #pragma omp parallel for collapse(4) schedule(static)
      for (int i = 0; i < nb_; ++i)
        for (int j = 0; j < nb_; ++j)
          for (int a = 0; a < vb_; ++a)
            for (int b = 0; b < vb_; ++b)
              Teff_bb(i, j, a, b) = (*t2_bb_dense)(i, j, a, b) + L2_bb_(i, j, a, b);
      #pragma omp parallel for collapse(4) schedule(static)
      for (int i = 0; i < na_; ++i)
        for (int j = 0; j < nb_; ++j)
          for (int a = 0; a < va_; ++a)
            for (int b = 0; b < vb_; ++b)
              Teff_ab(i, j, a, b) = (*t2_ab_dense)(i, j, a, b) + L2_ab_(i, j, a, b);

      TBLIS_VIEW_4D(t_Teff_bb, Teff_bb, nb_, nb_, vb_, vb_);
      TBLIS_VIEW_4D(t_Teff_ab, Teff_ab, na_, nb_, va_, vb_);

      Eigen::Tensor<double, 4> G_oooo_bb(nb_, nb_, nb_, nb_); G_oooo_bb.setZero();
      Eigen::Tensor<double, 4> G_oooo_ab(na_, na_, nb_, nb_); G_oooo_ab.setZero();
      TBLIS_VIEW_4D(t_Goooo_bb, G_oooo_bb, nb_, nb_, nb_, nb_);
      TBLIS_VIEW_4D(t_Goooo_ab, G_oooo_ab, na_, na_, nb_, nb_);
      tblis::mult<double>(0.125, t_T2_bb, "ijab", t_T2_bb, "mnab", 0.0, t_Goooo_bb, "minj");
      tblis::mult<double>(0.500, t_T2_ab, "ijab", t_T2_ab, "mnab", 0.0, t_Goooo_ab, "minj");

      Eigen::Tensor<double, 4> G_ovov_bb(nb_, vb_, nb_, vb_); G_ovov_bb.setZero();
      Eigen::Tensor<double, 4> G_oovv_bb(nb_, nb_, vb_, vb_); G_oovv_bb.setZero();
      TBLIS_VIEW_4D(t_Govov_bb, G_ovov_bb, nb_, vb_, nb_, vb_);
      TBLIS_VIEW_4D(t_Goovv_bb, G_oovv_bb, nb_, nb_, vb_, vb_);
      tblis::mult<double>( 0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Govov_bb, "iakc");
      tblis::mult<double>( 0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Govov_bb, "iakc");
      tblis::mult<double>(-0.5, t_T2_bb, "kjcb", t_T2_bb, "ijab", 0.0, t_Goovv_bb, "ikac");
      tblis::mult<double>(-0.5, t_T2_ab, "miea", t_T2_ab, "mkec", 1.0, t_Goovv_bb, "ikac");

      Eigen::Tensor<double, 4> G_ovov_ab(na_, va_, nb_, vb_); G_ovov_ab.setZero();
      Eigen::Tensor<double, 4> G_oovv_ab_ex(na_, na_, vb_, vb_); G_oovv_ab_ex.setZero();
      Eigen::Tensor<double, 4> G_oovv_ba_ex(nb_, nb_, va_, va_); G_oovv_ba_ex.setZero();
      TBLIS_VIEW_4D(t_Govov_ab, G_ovov_ab, na_, va_, nb_, vb_);
      TBLIS_VIEW_4D(t_Goovv_ab_ex, G_oovv_ab_ex, na_, na_, vb_, vb_);
      TBLIS_VIEW_4D(t_Goovv_ba_ex, G_oovv_ba_ex, nb_, nb_, va_, va_);
      tblis::mult<double>( 1.0, t_T2, "miea", t_T2_ab, "mkec", 0.0, t_Govov_ab, "iakc");
      tblis::mult<double>( 1.0, t_T2_bb, "kjcb", t_T2_ab, "ijab", 1.0, t_Govov_ab, "iakc");
      tblis::mult<double>(-0.5, t_T2_ab, "kmec", t_T2_ab, "imea", 0.0, t_Goovv_ab_ex, "ikac");
      tblis::mult<double>(-0.5, t_T2_ab, "mkce", t_T2_ab, "miae", 0.0, t_Goovv_ba_ex, "ikac");

      Eigen::Tensor<double, 4> V_ovvv_ab = integrals::ERITransformer::transform_finish_ket(half_ov_a, Cbv, Cbv, nbf_, na_, va_, vb_, vb_);
      Eigen::Tensor<double, 4> V_ooov_aa_bb = integrals::ERITransformer::transform_finish_ket(half_oo_a, Cbo, Cbv, nbf_, na_, na_, nb_, vb_);
      half_ov_a.resize(0, 0, 0, 0);
      half_oo_a.resize(0, 0, 0, 0);

      auto half_ov_b = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cbo, Cbv, nbf_, nb_, vb_);
      Eigen::Tensor<double, 4> V_ovvv_bb = integrals::ERITransformer::transform_finish_ket(half_ov_b, Cbv, Cbv, nbf_, nb_, vb_, vb_, vb_);
      Eigen::Tensor<double, 4> V_ovvv_ba = integrals::ERITransformer::transform_finish_ket(half_ov_b, Cav, Cav, nbf_, nb_, vb_, va_, va_);
      half_ov_b.resize(0, 0, 0, 0);

      auto half_oo_b = integrals::ERITransformer::transform_half_bra(eri_ao_ref, Cbo, Cbo, nbf_, nb_, nb_);
      Eigen::Tensor<double, 4> V_ooov_bb = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cbo, Cbv, nbf_, nb_, nb_, nb_, vb_);
      Eigen::Tensor<double, 4> V_ooov_bb_aa = integrals::ERITransformer::transform_finish_ket(half_oo_b, Cao, Cav, nbf_, nb_, nb_, na_, va_);
      half_oo_b.resize(0, 0, 0, 0);

      TBLIS_VIEW_4D(t_Vovvv_bb, V_ovvv_bb, nb_, vb_, vb_, vb_);
      TBLIS_VIEW_4D(t_Vooov_bb, V_ooov_bb, nb_, nb_, nb_, vb_);
      TBLIS_VIEW_4D(t_Vovvv_ab, V_ovvv_ab, na_, va_, vb_, vb_);
      TBLIS_VIEW_4D(t_Vovvv_ba, V_ovvv_ba, nb_, vb_, va_, va_);
      TBLIS_VIEW_4D(t_Vooov_aa_bb, V_ooov_aa_bb, na_, na_, nb_, vb_);
      TBLIS_VIEW_4D(t_Vooov_bb_aa, V_ooov_bb_aa, nb_, nb_, na_, va_);

      tblis::mult<double>( 1.0, t_Vovvv_ba, "kcab", t_Teff_ab, "ikbc", 1.0, t_Zmat_a, "ai");
      tblis::mult<double>(-1.0, t_Vooov_aa_bb, "jikc", t_Teff_ab, "jkac", 1.0, t_Zmat_a, "ai");

      tblis::mult<double>( 1.0, t_Vovvv_bb, "kcab", t_Teff_bb, "ikbc", 0.0, t_Zmat_b, "ai");
      tblis::mult<double>(-1.0, t_Vooov_bb, "jikc", t_Teff_bb, "jkac", 1.0, t_Zmat_b, "ai");
      tblis::mult<double>( 1.0, t_Vovvv_ab, "kcab", t_Teff_ab, "kicb", 1.0, t_Zmat_b, "ai");
      tblis::mult<double>(-1.0, t_Vooov_bb_aa, "jikc", t_Teff_ab, "kjca", 1.0, t_Zmat_b, "ai");

      Eigen::Tensor<double, 4> Y_ooov_bb(nb_, nb_, nb_, vb_); Y_ooov_bb.setZero();
      TBLIS_VIEW_4D(t_Yooov_bb, Y_ooov_bb, nb_, nb_, nb_, vb_);
      tblis::mult<double>( 1.0, t_T2_bb, "ijab", t_Vovvv_bb, "mafb", 0.0, t_Yooov_bb, "ijmf");
      tblis::mult<double>(-0.5, t_T2_bb, "ijxf", t_Yooov_bb, "ijmf", 1.0, t_Zmat_b, "xm");

      Eigen::Tensor<double, 4> Y_ooov_ab_a(na_, nb_, na_, vb_); Y_ooov_ab_a.setZero();
      TBLIS_VIEW_4D(t_Yooov_ab_a, Y_ooov_ab_a, na_, nb_, na_, vb_);
      tblis::mult<double>( 1.0, t_T2_ab, "ijab", t_Vovvv_ab, "mafb", 0.0, t_Yooov_ab_a, "ijmf");
      tblis::mult<double>(-1.0, t_T2_ab, "ijxf", t_Yooov_ab_a, "ijmf", 1.0, t_Zmat_a, "xm");

      Eigen::Tensor<double, 4> Y_ooov_ab_b(na_, nb_, nb_, va_); Y_ooov_ab_b.setZero();
      TBLIS_VIEW_4D(t_Yooov_ab_b, Y_ooov_ab_b, na_, nb_, nb_, va_);
      tblis::mult<double>( 1.0, t_T2_ab, "ijab", t_Vovvv_ba, "mbea", 0.0, t_Yooov_ab_b, "ijme");
      tblis::mult<double>(-1.0, t_T2_ab, "ijex", t_Yooov_ab_b, "ijme", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>(1.0, t_Goooo_ab, "minj", t_Vooov_bb_aa, "njix", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(1.0, t_Goooo_ab, "imnj", t_Vooov_bb_aa, "njix", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(1.0, t_Goooo_ab, "inmj", t_Vooov_aa_bb, "injx", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_ab, "injm", t_Vooov_aa_bb, "injx", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>(1.0, t_Goooo_bb, "minj", t_Vooov_bb, "njix", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_bb, "imnj", t_Vooov_bb, "njix", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_bb, "inmj", t_Vooov_bb, "injx", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(1.0, t_Goooo_bb, "injm", t_Vooov_bb, "injx", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>( 1.0, t_Govov_bb, "makc", t_Vovvv_bb, "kcxa", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Govov_bb, "ixkc", t_Vooov_bb, "imkc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Govov_bb, "iamc", t_Vovvv_bb, "iaxc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Govov_bb, "iakx", t_Vooov_bb, "kmia", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>( 1.0, t_Goovv_bb, "mkac", t_Vovvv_bb, "kxca", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Goovv_bb, "imac", t_Vovvv_bb, "ixac", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_bb, "ikxc", t_Vooov_bb, "ikmc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_bb, "ikax", t_Vooov_bb, "ikma", 1.0, t_Zmat_b, "xm");

      tblis::mult<double>( 1.0, t_Govov_ab, "makc", t_Vovvv_ba, "kcxa", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Govov_ab, "ixkc", t_Vooov_aa_bb, "imkc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ab_ex, "mkac", t_Vovvv_ab, "kxac", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ab_ex, "imac", t_Vovvv_ab, "ixac", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikxc", t_Vooov_bb_aa, "ikmc", 1.0, t_Zmat_a, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ba_ex, "ikax", t_Vooov_bb_aa, "ikma", 1.0, t_Zmat_a, "xm");

      tblis::mult<double>( 1.0, t_Govov_ab, "iamc", t_Vovvv_ab, "iaxc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Govov_ab, "iakx", t_Vooov_bb_aa, "kmia", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ba_ex, "mkac", t_Vovvv_ba, "kxac", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>( 1.0, t_Goovv_ba_ex, "imac", t_Vovvv_ba, "ixac", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikxc", t_Vooov_aa_bb, "ikmc", 1.0, t_Zmat_b, "xm");
      tblis::mult<double>(-1.0, t_Goovv_ab_ex, "ikax", t_Vooov_aa_bb, "ikma", 1.0, t_Zmat_b, "xm");
    } else if (is_restricted) {
      Z_mat_b = Z_mat_a;
    }
  }

  Eigen::MatrixXd G_full_a = Eigen::MatrixXd::Zero(nbf_, nbf_);
  G_full_a.block(0, 0, na_, na_) = G_oo_alpha_;
  G_full_a.block(na_, na_, va_, va_) = G_vv_alpha_;
  Eigen::MatrixXd P_corr_a = scf_.C_alpha * G_full_a * scf_.C_alpha.transpose();

  Eigen::MatrixXd G_full_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
  Eigen::MatrixXd P_corr_b = Eigen::MatrixXd::Zero(nbf_, nbf_);

  if (is_restricted) {
    G_full_b = G_full_a;
    P_corr_b = P_corr_a;
  } else if (nb_ > 0) {
    G_full_b.block(0, 0, nb_, nb_) = G_oo_beta_;
    G_full_b.block(nb_, nb_, vb_, vb_) = G_vv_beta_;
    P_corr_b = scf_.C_beta * G_full_b * scf_.C_beta.transpose();
  }

  const Eigen::MatrixXd& F_HF_ao_a = scf_.F_alpha;
  const Eigen::MatrixXd& F_HF_ao_b = scf_.F_beta;
  Eigen::MatrixXd F_HF_mo_a = scf_.C_alpha.transpose() * F_HF_ao_a * scf_.C_alpha;

  Eigen::MatrixXd F_HF_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
  if (nb_ > 0) F_HF_mo_b = scf_.C_beta.transpose() * F_HF_ao_b * scf_.C_beta;

  Eigen::MatrixXd G_gamma_ao_a, G_gamma_ao_b;
  build_fock_fast(P_corr_a, P_corr_b, G_gamma_ao_a, G_gamma_ao_b);
  G_gamma_ao_a -= H_core_;
  if (nb_ > 0) G_gamma_ao_b -= H_core_;

  Eigen::MatrixXd G_gamma_mo_a = scf_.C_alpha.transpose() * G_gamma_ao_a * scf_.C_alpha;
  Eigen::MatrixXd G_gamma_mo_b = Eigen::MatrixXd::Zero(nbf_, nbf_);
  if (nb_ > 0) G_gamma_mo_b = scf_.C_beta.transpose() * G_gamma_ao_b * scf_.C_beta;

  F_gen_a_ = F_HF_mo_a + G_gamma_mo_a;
  if (na_ > 0 && va_ > 0) {
    Eigen::MatrixXd F_HF_vo_a = F_HF_mo_a.block(na_, 0, va_, na_);
    Eigen::MatrixXd L_sep_a = F_HF_vo_a * G_oo_alpha_ - G_vv_alpha_ * F_HF_vo_a;
    F_gen_a_.block(na_, 0, va_, na_) += L_sep_a + Z_mat_a;
    F_gen_a_.block(0, na_, na_, va_) += (L_sep_a + Z_mat_a).transpose();
  }

  F_gen_b_ = F_HF_mo_b + G_gamma_mo_b;
  if (nb_ > 0 && vb_ > 0) {
    Eigen::MatrixXd F_HF_vo_b = F_HF_mo_b.block(nb_, 0, vb_, nb_);
    Eigen::MatrixXd L_sep_b = F_HF_vo_b * G_oo_beta_ - G_vv_beta_ * F_HF_vo_b;
    F_gen_b_.block(nb_, 0, vb_, nb_) += L_sep_b + Z_mat_b;
    F_gen_b_.block(0, nb_, nb_, vb_) += (L_sep_b + Z_mat_b).transpose();
  } else if (is_restricted) {
    F_gen_b_ = F_gen_a_;
  }
}"""

    src = replace_cpp_function(src, r"void\s+OMP3::build_opdm_beta\s*\(\s*\)", new_opdm_beta)
    src = replace_cpp_function(src, r"void\s+OMP3::build_generalized_fock\s*\(\s*\)", new_build_gen_fock)
    return src, "PATCHED (std::move L2 + streaming OPDM beta + active-sliced eager-free OMP3 Fock)"


# ==============================================================================
# PATCH 10: src/gradient/hpc_geom_opt.cc
# ==============================================================================

def patch_hpc_geom_opt_cc(src: str) -> Tuple[str, str]:
    marker = "// [INJECTED_HPC_V5_GEOM_FC_MEM]"
    if marker in src:
        return src, "SKIP (sudah terpasang)"

    pat_sup = r'return\s*\(method\s*==\s*"scf"\s*\|\|\s*method\s*==\s*"hf"\s*\|\|\s*method\s*==\s*"rhf"\s*\|\|\s*method\s*==\s*"uhf"\s*\|\|\s*method\s*==\s*"rohf"\s*\|\|\s*method\s*==\s*"omp2"\);'
    rep_sup = (
        f"{marker}\n"
        '  if (method == "omp2" && (cfg.freeze_core || cfg.n_frozen > 0)) return false;\n'
        '  return (method == "scf" || method == "hf" || method == "rhf" || method == "uhf" || method == "rohf" || method == "omp2");'
    )
    src, n_sup = sub_literal(pat_sup, rep_sup, src, count=1)
    if n_sup != 1:
        raise RuntimeError("Anchor supports_analytic_df_gradient tidak ditemukan di src/gradient/hpc_geom_opt.cc")

    pat_scfg = r"scf_conf\.max_iterations\s*=\s*200;"
    rep_scfg = (
        "scf_conf.max_iterations = 200;\n"
        "  scf_conf.memory_limit_gb = cfg.memory_limit_gb;"
    )
    src, n_scfg = sub_literal(pat_scfg, rep_scfg, src, count=2)
    if n_scfg < 1:
        raise RuntimeError("Anchor scf_conf.max_iterations tidak ditemukan di src/gradient/hpc_geom_opt.cc")

    pat_mcfg = r"mp2_conf\.max_iterations\s*=\s*100;"
    rep_mcfg = (
        "mp2_conf.max_iterations = 100;\n"
        "  mp2_conf.freeze_core = cfg.freeze_core;\n"
        "  mp2_conf.n_frozen = cfg.n_frozen;\n"
        "  mp2_conf.memory_limit_gb = cfg.memory_limit_gb;"
    )
    src, n_mcfg = sub_literal(pat_mcfg, rep_mcfg, src, count=2)
    if n_mcfg < 1:
        raise RuntimeError("Anchor mp2_conf.max_iterations tidak ditemukan di src/gradient/hpc_geom_opt.cc")

    return src, f"PATCHED (DF-OMP2 frozen-core FD guard + forward freeze_core/memory_limit_gb [{n_scfg}/{n_mcfg}])"


PATCH_PIPELINE = [
    ("include/mshqc/scf/scf.h", patch_scf_h),
    ("include/mshqc/mp2/mp2.h", patch_mp2_h),
    ("include/mshqc/gradient/hpc_geom_opt.h", patch_hpc_geom_opt_h),
    ("python/bindings.cc", patch_bindings_cc),
    ("src/utils/hdf5_io.cc", patch_hdf5_io_cc),
    ("src/ints/integrals.cc", patch_integrals_cc),
    ("src/scf/scf.cc", patch_scf_cc),
    ("src/mp2/mp2.cc", patch_mp2_cc),
    ("src/mp3/mp3.cc", patch_mp3_cc),
    ("src/gradient/hpc_geom_opt.cc", patch_hpc_geom_opt_cc),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="MSHQC-Sync HPC Heavy-Molecule Production Injector v5")
    parser.add_argument("--root", type=str, default=".", help="Path root repositori mshqc-sync")
    parser.add_argument("--dry-run", action="store_true", help="Uji pencocokan & integritas tanpa menulis file")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / f".backup_hpc_v5_{ts}"

    print("=" * 94)
    print(f" MSHQC-SYNC HPC HEAVY-MOLECULE PRODUCTION INJECTOR V5 | Root: {root}")
    print(f" Mode: {'DRY-RUN (no write)' if args.dry_run else 'APPLY (auto-backup + C++ integrity check)'}")
    print("=" * 94)

    patched_count = 0
    for rel_path, fn in PATCH_PIPELINE:
        fpath = root / rel_path
        if not fpath.is_file():
            print(f"[MISS] {rel_path:<38s} -> file tidak ditemukan!")
            return 1

        orig_text = fpath.read_text(encoding="utf-8")
        try:
            new_text, status = fn(orig_text)
            verify_cpp_integrity(new_text, rel_path)
        except Exception as exc:
            print(f"[FAIL] {rel_path:<38s} -> {exc}")
            return 1

        changed = (new_text != orig_text)
        print(f"[OK  ] {rel_path:<38s} -> {status}")

        if changed and not args.dry_run:
            bpath = backup_root / rel_path
            bpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, bpath)
            fpath.write_text(new_text, encoding="utf-8")
            patched_count += 1

    print("=" * 94)
    if args.dry_run:
        print("DRY-RUN SELESAI: Seluruh 10 file lolos pencocokan anchor & verifikasi sintaks kutip/brace.")
    else:
        print(f"SELESAI: {patched_count} file diperbarui & terverifikasi. Backup: {backup_root}")
        print("Langkah selanjutnya: rebuild C++20 -> pytest -> verify_audit_and_scaling -> heavy_large.")
    return 0


if __name__ == "__main__":
    sys.exit(main())