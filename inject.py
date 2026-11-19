#!/usr/bin/env python3
"""
inject_audit_fixes.py
---------------------
Injector terpadu untuk:
1. [BUG FIX - SEGFAULT] src/symmetry/molecule_sym.cc
2. [BUG FIX + OPT]      src/gradient/hpc_geom_opt.cc (DF-ROHF analytic grad, Warm-Start SCF/OMP, O(N) Analytic-Grad Hessian)
3. [OPT + FIX]          src/mp/omp_gradient.cc (O(O^2 V) SOSCF/CPHF contraction & zero realloc)
4. [GUARD FIX]          src/mp3/mp3.cc (UMP3 frozen-core guard)
5. [OPT]                src/integrals/cholesky_eri.cc (single-pass packed HDF5 load)
6. [OPT]                src/ints/integrals.cc (exact density-block screening on DF 2e gradient)
7. [CLEANUP]            src/scf/scf.cc (PetiteList schwarz fallback & skip unused FockBuilder in DF/CD)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY-RUN] Akan memperbarui {path}")
        return
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(content, encoding="utf-8")
    print(f"  [OK] Berhasil memperbarui {path} (backup: {bak.name})")


def find_matching_brace(src: str, open_idx: int) -> int:
    """Temukan indeks '}' pasangan dari '{' pada open_idx (mengabaikan string/komentar sederhana)."""
    depth = 0
    i = open_idx
    n = len(src)
    in_line_comment = False
    in_block_comment = False
    in_str = False
    in_char = False
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif in_char:
            if ch == "\\":
                i += 1
            elif ch == "'":
                in_char = False
        else:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif ch == '"':
                in_str = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError("Tidak menemukan pasangan '}' yang cocok.")


def replace_function_body(src: str, sig_regex: str, new_func: str) -> str:
    m = re.search(sig_regex, src, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Signature tidak ditemukan: {sig_regex}")
    brace_open = src.find("{", m.end() - 1)
    if brace_open == -1:
        raise RuntimeError(f"Kurung kurawal buka tidak ditemukan untuk: {sig_regex}")
    brace_close = find_matching_brace(src, brace_open)
    return src[: m.start()] + new_func + src[brace_close + 1 :]


# =============================================================================
# 1. FIX SEGFAULT: src/symmetry/molecule_sym.cc
# =============================================================================
def patch_molecule_sym(root: Path, dry_run: bool) -> None:
    p = root / "src/symmetry/molecule_sym.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_NO_RAO_WRITE]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    pattern = re.compile(
        r"^[ \t]*R_ao_\[k\]\(start_j\s*\+\s*d,\s*start_i\s*\+\s*d\)\s*=\s*phase;[ \t]*\r?\n",
        flags=re.MULTILINE,
    )
    if not pattern.search(src):
        print(f"[SKIP] Baris segfault R_ao_[k] tidak ditemukan di {p} (mungkin sudah bersih).")
        return

    new_src = pattern.sub(f"          {marker}\n", src, count=1)
    write_text(p, new_src, dry_run)


# =============================================================================
# 2. OPT + FIX: src/mp/omp_gradient.cc (O(O^2 V) contraction, zero realloc)
# =============================================================================
def patch_omp_gradient(root: Path, dry_run: bool) -> None:
    p = root / "src/mp/omp_gradient.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_SOSCF_O2V]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    # 2a. Deklarasi buffer di compute_soscf_step
    old_decl = re.compile(
        r"Eigen::MatrixXd\s+T_oo_a\(na_,\s*va_\),\s*T_vo_a\(va_,\s*na_\);\s*\n"
        r"([ \t]*)Eigen::MatrixXd\s+T_oo_b,\s*T_vo_b;\s*\n"
        r"([ \t]*)if\s*\(!is_restricted\s*&&\s*dim_b\s*>\s*0\)\s*\{\s*\n"
        r"[ \t]*T_oo_b\.resize\(nb_,\s*vb_\);\s*\n"
        r"[ \t]*T_vo_b\.resize\(vb_,\s*nb_\);\s*\n"
        r"[ \t]*\}",
        flags=re.MULTILINE,
    )
    if not old_decl.search(src):
        raise RuntimeError(f"Anchor deklarasi T_vo_a/T_vo_b tidak ditemukan di {p}")

    src = old_decl.sub(
        f"{marker}\n"
        r"      Eigen::MatrixXd T_oo_a(na_, va_), T_oo2_a(na_, na_);" + "\n"
        r"\1Eigen::MatrixXd T_oo_b, T_oo2_b;" + "\n"
        r"\2if (!is_restricted && dim_b > 0) {" + "\n"
        r"\2  T_oo_b.resize(nb_, vb_);" + "\n"
        r"\2  T_oo2_b.resize(nb_, nb_);" + "\n"
        r"\2}",
        src,
        count=1,
    )

    # 2b. Kontraksi alpha di compute_soscf_step
    old_ka = re.compile(
        r"T_vo_a\.noalias\(\)\s*=\s*Bvo_a\s*\*\s*kappa_a;\s*\n"
        r"[ \t]*Ka_tls\[tid\]\.noalias\(\)\s*\+=\s*\(T_vo_a\s*\*\s*Bvo_a\)\.transpose\(\);",
        flags=re.MULTILINE,
    )
    if not old_ka.search(src):
        raise RuntimeError(f"Anchor T_vo_a di compute_soscf_step tidak ditemukan di {p}")
    src = old_ka.sub(
        "T_oo2_a.noalias() = kappa_a * Bvo_a;\n"
        "          Ka_tls[tid].noalias() += T_oo2_a.transpose() * Bvo_a.transpose();",
        src,
        count=1,
    )

    # 2c. Kontraksi beta di compute_soscf_step
    old_kb = re.compile(
        r"T_vo_b\.noalias\(\)\s*=\s*Bvo_b\s*\*\s*kappa_b;\s*\n"
        r"[ \t]*Kb_tls\[tid\]\.noalias\(\)\s*\+=\s*\(T_vo_b\s*\*\s*Bvo_b\)\.transpose\(\);",
        flags=re.MULTILINE,
    )
    if not old_kb.search(src):
        raise RuntimeError(f"Anchor T_vo_b di compute_soscf_step tidak ditemukan di {p}")
    src = old_kb.sub(
        "T_oo2_b.noalias() = kappa_b * Bvo_b;\n"
        "          Kb_tls[tid].noalias() += T_oo2_b.transpose() * Bvo_b.transpose();",
        src,
        count=1,
    )

    # 2d. Kontraksi di solve_cphf_pcg_ov
    old_cphf = re.compile(
        r"Eigen::MatrixXd\s+T_oo\(na,\s*va\),\s*T_vo\(va,\s*na\);",
        flags=re.MULTILINE,
    )
    old_cphf_mul = re.compile(
        r"T_vo\.noalias\(\)\s*=\s*Bvo\s*\*\s*kappa_ov;\s*\n"
        r"[ \t]*K_tls\[tid\]\.noalias\(\)\s*\+=\s*T_vo\s*\*\s*Bvo;",
        flags=re.MULTILINE,
    )
    if old_cphf.search(src) and old_cphf_mul.search(src):
        src = old_cphf.sub("Eigen::MatrixXd T_oo(na, va), T_oo2(na, na);", src, count=1)
        src = old_cphf_mul.sub(
            "T_oo2.noalias() = kappa_ov * Bvo;\n"
            "          K_tls[tid].noalias() += Bvo * T_oo2;",
            src,
            count=1,
        )

    write_text(p, src, dry_run)


# =============================================================================
# 3. BUG FIX (DF-ROHF) + WARM-START SCF/OMP + O(N) ANALYTIC FREQ: hpc_geom_opt.cc
# =============================================================================
def patch_hpc_geom_opt(root: Path, dry_run: bool) -> None:
    p = root / "src/gradient/hpc_geom_opt.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_ROHF_WARMSTART_FREQ]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    # 3a. Tambahkan struct WarmStartGuess & deklarasi evaluate_single_point_with_guess setelah GradCurvatureResult
    anchor_struct = re.compile(
        r"(struct\s+GradCurvatureResult\s*\{[^}]*\};)",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not anchor_struct.search(src):
        raise RuntimeError(f"struct GradCurvatureResult tidak ditemukan di {p}")

    warmstart_decl = r"""\1

// [INJECTED_AUDIT_FIX_ROHF_WARMSTART_FREQ]
struct WarmStartGuess {
  bool has_scf = false;
  bool has_omp = false;
  Eigen::MatrixXd scf_C_alpha;
  Eigen::MatrixXd scf_C_beta;
  Eigen::MatrixXd omp_C_alpha;
  Eigen::MatrixXd omp_C_beta;
};

double evaluate_single_point_with_guess(
    const Molecule& mol,
    const GeomOptConfig& cfg,
    const WarmStartGuess* guess_in,
    WarmStartGuess* guess_out);
"""
    src = anchor_struct.sub(warmstart_decl, src, count=1)

    # 3b. Ganti compute_analytic_df_gradient_internal (Fix DF-ROHF + Warm-Start)
    new_analytic_fn = r"""GradCurvatureResult compute_analytic_df_gradient_internal(
    const Molecule& mol,
    const GeomOptConfig& cfg,
    const WarmStartGuess* guess_in = nullptr,
    WarmStartGuess* guess_out = nullptr) {

  ScopedCoutSilencer silencer(true);
  const int natoms = static_cast<int>(mol.n_atoms());
  const int ndof = 3 * natoms;

  GradCurvatureResult out;
  out.grad = Eigen::VectorXd::Zero(ndof);
  out.diag_curv = Eigen::VectorXd::Zero(ndof);
  out.has_curv = Eigen::VectorXi::Zero(ndof);

  std::string method = to_lower_copy(cfg.method);
  if (method.rfind("df-", 0) == 0 || method.rfind("df_", 0) == 0) method = method.substr(3);

  const std::string basis_name = cfg.basis_name.empty() ? "cc-pVDZ" : cfg.basis_name;
  const std::string aux_name = cfg.aux_basis_name.empty() ? (basis_name + "-RI") : cfg.aux_basis_name;

  BasisSet basis(basis_name, mol);
  auto integrals = std::make_shared<IntegralEngine>(mol, basis);
  const int nbf = basis.n_basis_functions();

  const int n_elec = mol.n_electrons();
  const int mult = mol.multiplicity();
  const int na = (n_elec + mult - 1) / 2;
  const int nb = n_elec - na;

  std::string ref = to_lower_copy(cfg.reference);
  if (ref.empty() || ref == "auto") {
    if (method == "rhf") ref = "rhf";
    else if (method == "rohf") ref = "rohf";
    else if (method == "uhf") ref = "uhf";
    else ref = (mult == 1) ? "rhf" : "uhf";
  }

  SCFConfig scf_conf;
  scf_conf.scf_type = "incore";
  scf_conf.eri_method = "df";
  scf_conf.use_df = true;
  scf_conf.aux_basis_name = aux_name;
  scf_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;
  scf_conf.density_threshold = scf_conf.energy_threshold;
  scf_conf.df_threshold = std::min(1.0e-12, scf_conf.energy_threshold);
  scf_conf.print_level = 0;
  scf_conf.max_iterations = 200;

  if (guess_in && guess_in->has_scf && guess_in->scf_C_alpha.rows() == nbf) {
    scf_conf.has_initial_guess = true;
    scf_conf.init_C_alpha = guess_in->scf_C_alpha;
    if (guess_in->scf_C_beta.rows() == nbf) {
      scf_conf.init_C_beta = guess_in->scf_C_beta;
    }
  }

  SCFResult scf_res;
  if (ref == "rhf") {
    RHF solver(mol, basis, integrals, scf_conf);
    scf_res = solver.compute();
  } else if (ref == "rohf") {
    ROHF solver(mol, basis, integrals, na, nb, scf_conf);
    scf_res = solver.compute();
  } else {
    UHF solver(mol, basis, integrals, na, nb, scf_conf);
    scf_res = solver.compute();
  }

  if (guess_out) {
    guess_out->scf_C_alpha = scf_res.C_alpha;
    guess_out->scf_C_beta = scf_res.C_beta;
    guess_out->has_scf = (scf_res.C_alpha.rows() == nbf);
  }

  Eigen::MatrixXd P_eff_ao, W_eff_ao, Gamma_3c, gamma_2c;
  const int n_aux = static_cast<int>(scf_res.L_mat.cols());

  if (method == "omp2") {
    MP2Config mp2_conf;
    mp2_conf.scf_type = "incore";
    mp2_conf.eri_method = "df";
    mp2_conf.use_df = true;
    mp2_conf.aux_basis_name = aux_name;
    mp2_conf.energy_threshold = scf_conf.energy_threshold;
    mp2_conf.df_threshold = scf_conf.df_threshold;
    mp2_conf.gradient_threshold = (cfg.orb_grad_thresh > 0.0) ? std::min(1.0e-10, cfg.orb_grad_thresh) : 1.0e-10;
    mp2_conf.opt_method = cfg.opt_method;
    mp2_conf.print_level = 0;
    mp2_conf.max_iterations = 100;

    if (guess_in && guess_in->has_omp && guess_in->omp_C_alpha.rows() == nbf) {
      mp2_conf.has_initial_guess = true;
      mp2_conf.init_C_alpha = guess_in->omp_C_alpha;
      if (guess_in->omp_C_beta.rows() == nbf) {
        mp2_conf.init_C_beta = guess_in->omp_C_beta;
      }
    }

    OMP2 omp2_solver(mol, basis, integrals, scf_res, mp2_conf);
    MP2Result omp2_res = omp2_solver.compute();
    out.energy = omp2_res.energy_total;

    if (guess_out) {
      guess_out->omp_C_alpha = omp2_res.C_alpha;
      guess_out->omp_C_beta = omp2_res.C_beta;
      guess_out->has_omp = (omp2_res.C_alpha.rows() == nbf);
    }

    OMP2GradientData gd = omp2_solver.build_df_gradient_data();
    P_eff_ao = std::move(gd.P_eff_ao);
    W_eff_ao = std::move(gd.W_eff_ao);
    Gamma_3c = std::move(gd.Gamma_3c);
    gamma_2c = std::move(gd.gamma_2c);
  } else {
    out.energy = scf_res.energy_total;
    const bool is_rhf = (ref == "rhf");
    const Eigen::MatrixXd& Pa = scf_res.P_alpha;
    const Eigen::MatrixXd& Pb = is_rhf ? scf_res.P_alpha : scf_res.P_beta;
    P_eff_ao = Pa + Pb;

    const Eigen::MatrixXd Ca_o = scf_res.C_alpha.leftCols(na);
    if (ref == "rohf") {
      const int n_c = nb;
      const int n_o = na - nb;
      const Eigen::MatrixXd& C = scf_res.C_alpha;
      Eigen::MatrixXd Fa_mo = C.transpose() * scf_res.F_alpha * C;
      Eigen::MatrixXd Fb_mo = C.transpose() * scf_res.F_beta * C;
      Eigen::MatrixXd W_mo = Eigen::MatrixXd::Zero(nbf, nbf);
      if (n_c > 0) {
        W_mo.topLeftCorner(n_c, n_c) = Fa_mo.topLeftCorner(n_c, n_c) + Fb_mo.topLeftCorner(n_c, n_c);
      }
      if (n_o > 0) {
        W_mo.block(n_c, n_c, n_o, n_o) = Fa_mo.block(n_c, n_c, n_o, n_o);
      }
      if (n_c > 0 && n_o > 0) {
        Eigen::MatrixXd W_co = Fa_mo.block(0, n_c, n_c, n_o) + 0.5 * Fb_mo.block(0, n_c, n_c, n_o);
        W_mo.block(0, n_c, n_c, n_o) = W_co;
        W_mo.block(n_c, 0, n_o, n_c) = W_co.transpose();
      }
      W_eff_ao = C * W_mo * C.transpose();
    } else {
      W_eff_ao = Ca_o * scf_res.orbital_energies_alpha.head(na).asDiagonal() * Ca_o.transpose();
      if (is_rhf) {
        W_eff_ao *= 2.0;
      } else if (nb > 0) {
        const Eigen::MatrixXd Cb_o = scf_res.C_beta.leftCols(nb);
        W_eff_ao.noalias() += Cb_o * scf_res.orbital_energies_beta.head(nb).asDiagonal() * Cb_o.transpose();
      }
    }

    const bool is_packed = integrals::is_packed_ao_matrix(scf_res.L_mat, nbf);
    Eigen::VectorXd c_vec = integrals::contract_ao_factor_with_sym(scf_res.L_mat, P_eff_ao, nbf);
    Gamma_3c.resize(static_cast<Eigen::Index>(nbf) * nbf, n_aux);

    const Eigen::MatrixXd Cb_o = (!is_rhf && nb > 0) ? scf_res.C_beta.leftCols(nb) : Eigen::MatrixXd();
    #pragma omp parallel
    {
      Eigen::MatrixXd G_P(nbf, nbf), T_a(nbf, na), T_b, B_P_buf;
      if (!is_rhf && nb > 0) T_b.resize(nbf, nb);
      if (is_packed) B_P_buf.resize(nbf, nbf);

      #pragma omp for schedule(static) nowait
      for (int P = 0; P < n_aux; ++P) {
        const double* col_p = scf_res.L_mat.col(P).data();
        if (is_packed) integrals::unpack_ao_column(col_p, nbf, true, B_P_buf);
        Eigen::Map<const Eigen::MatrixXd> B_P(is_packed ? B_P_buf.data() : col_p, nbf, nbf);

        G_P.noalias() = P_eff_ao * c_vec(P);
        T_a.noalias() = B_P * Ca_o;
        Eigen::MatrixXd Boo_a = Ca_o.transpose() * T_a;
        G_P.noalias() -= (is_rhf ? 2.0 : 1.0) * (Ca_o * Boo_a * Ca_o.transpose());
        if (!is_rhf && nb > 0) {
          T_b.noalias() = B_P * Cb_o;
          Eigen::MatrixXd Boo_b = Cb_o.transpose() * T_b;
          G_P.noalias() -= Cb_o * Boo_b * Cb_o.transpose();
        }
        Gamma_3c.col(P) = Eigen::Map<const Eigen::VectorXd>(G_P.data(), nbf * nbf);
      }
    }

    gamma_2c = integrals::compute_gamma2c_from_full_and_L(Gamma_3c, scf_res.L_mat, nbf);
  }

  BasisSet aux_basis(aux_name, mol);
  BasisSet combined = basis;
  combined.append(aux_basis);
  auto df_ints = std::make_shared<IntegralEngine>(mol, combined);
  Eigen::MatrixXd J_inv_sqrt = compute_df_metric_inv_sqrt(basis, aux_basis, df_ints, scf_conf.df_threshold);

  Eigen::MatrixXd Gamma_tilde(static_cast<Eigen::Index>(nbf) * nbf, n_aux);
  constexpr int tile_m = 256;
  #pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < nbf * nbf; r += tile_m) {
    int rows = std::min(tile_m, nbf * nbf - r);
    Gamma_tilde.middleRows(r, rows).noalias() = Gamma_3c.middleRows(r, rows) * J_inv_sqrt;
  }

  Eigen::MatrixXd tmp_g(n_aux, n_aux), gamma_tilde(n_aux, n_aux);
  constexpr int tile_a = 64;
  #pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < n_aux; r += tile_a) {
    int rows = std::min(tile_a, n_aux - r);
    tmp_g.middleRows(r, rows).noalias() = J_inv_sqrt.middleRows(r, rows) * gamma_2c;
  }
  #pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < n_aux; r += tile_a) {
    int rows = std::min(tile_a, n_aux - r);
    gamma_tilde.middleRows(r, rows).noalias() = tmp_g.middleRows(r, rows) * J_inv_sqrt;
  }

  Eigen::MatrixXd g_tot = compute_nuclear_repulsion_gradient(mol)
      + integrals->compute_1e_gradient(P_eff_ao, W_eff_ao)
      + df_ints->compute_df_2e_gradient(
          basis.n_shells(), aux_basis.n_shells(),
          nbf, aux_basis.n_basis_functions(),
          Gamma_tilde, gamma_tilde);

  Eigen::VectorXd x0(ndof);
  std::vector<double> masses(natoms);
  for (int a = 0; a < natoms; ++a) {
    x0(3 * a + 0) = mol.atom(a).x;
    x0(3 * a + 1) = mol.atom(a).y;
    x0(3 * a + 2) = mol.atom(a).z;
    masses[a] = get_exact_isotopic_mass(mol.atom(a).atomic_number);
    out.grad(3 * a + 0) = g_tot(a, 0);
    out.grad(3 * a + 1) = g_tot(a, 1);
    out.grad(3 * a + 2) = g_tot(a, 2);
  }
  if (cfg.project_trans_rot) {
    out.grad = HPCGeometryOptimizer::project_out_trans_rot(x0, out.grad, masses);
  }
  return out;
}"""

    src = replace_function_body(
        src,
        r"GradCurvatureResult\s+compute_analytic_df_gradient_internal\s*\([^)]*\)\s*\{",
        new_analytic_fn,
    )

    # 3c. Ganti compute_gradient_and_curvature_internal agar mendukung Warm-Start ke x0 +- h
    new_gc_fn = r"""GradCurvatureResult compute_gradient_and_curvature_internal(
    const Molecule& mol,
    const GeomOptConfig& cfg,
    WarmStartGuess* guess_io = nullptr) {

  if (supports_analytic_df_gradient(cfg)) {
    return compute_analytic_df_gradient_internal(mol, cfg, guess_io, guess_io);
  }

  const int natoms = static_cast<int>(mol.n_atoms());
  const int ndof = 3 * natoms;
  Eigen::VectorXd x0(ndof);
  std::vector<double> masses(natoms);
  for (int i = 0; i < natoms; ++i) {
    x0(3 * i + 0) = mol.atom(i).x;
    x0(3 * i + 1) = mol.atom(i).y;
    x0(3 * i + 2) = mol.atom(i).z;
    masses[i] = get_exact_isotopic_mass(mol.atom(i).atomic_number);
  }

  WarmStartGuess center_guess;
  GradCurvatureResult out;
  out.energy = evaluate_single_point_with_guess(mol, cfg, guess_io, &center_guess);
  if (guess_io) *guess_io = center_guess;

  out.grad = Eigen::VectorXd::Zero(ndof);
  out.diag_curv = Eigen::VectorXd::Zero(ndof);
  out.has_curv = Eigen::VectorXi::Zero(ndof);

  if (natoms <= 1) {
    return out;
  }

  const double h = (cfg.fd_step_bohr > 0.0) ? cfg.fd_step_bohr : 5.0e-4;

  for (int a = 0; a < natoms - 1; ++a) {
    for (int xyz = 0; xyz < 3; ++xyz) {
      bool all_on_mirror_plane = true;
      for (int b = 0; b < natoms; ++b) {
        if (std::abs(x0(3 * b + xyz)) > 1.0e-11) {
          all_on_mirror_plane = false;
          break;
        }
      }
      if (all_on_mirror_plane) {
        out.grad(3 * a + xyz) = 0.0;
        continue;
      }

      Eigen::VectorXd xp = x0;
      Eigen::VectorXd xm = x0;
      xp(3 * a + xyz) += h;
      xm(3 * a + xyz) -= h;

      const double ep = evaluate_single_point_with_guess(
          make_displaced_molecule(mol, xp), cfg, &center_guess, nullptr);
      const double em = evaluate_single_point_with_guess(
          make_displaced_molecule(mol, xm), cfg, &center_guess, nullptr);

      out.grad(3 * a + xyz) = (ep - em) / (2.0 * h);
      const double curv = (ep - 2.0 * out.energy + em) / (h * h);
      if (std::isfinite(curv) && curv > 1.0e-2 && curv < 20.0) {
        out.diag_curv(3 * a + xyz) = curv;
        out.has_curv(3 * a + xyz) = 1;
      }
    }
  }

  for (int xyz = 0; xyz < 3; ++xyz) {
    double sum_other = 0.0;
    for (int a = 0; a < natoms - 1; ++a) {
      sum_other += out.grad(3 * a + xyz);
    }
    out.grad(3 * (natoms - 1) + xyz) = -sum_other;
  }

  if (cfg.project_trans_rot) {
    out.grad = HPCGeometryOptimizer::project_out_trans_rot(x0, out.grad, masses);
  }

  return out;
}

double evaluate_single_point_with_guess(
    const Molecule& mol,
    const GeomOptConfig& cfg,
    const WarmStartGuess* guess_in,
    WarmStartGuess* guess_out) {

  ScopedCoutSilencer silencer(true);

  std::string method = to_lower_copy(cfg.method);
  std::string eri_mode = to_lower_copy(cfg.eri_method);
  if (eri_mode == "ri") eri_mode = "df";

  if (method.rfind("df-", 0) == 0 || method.rfind("df_", 0) == 0) {
    eri_mode = "df";
    method = method.substr(3);
  } else if (method.rfind("cholesky-", 0) == 0 || method.rfind("cholesky_", 0) == 0) {
    eri_mode = "cholesky";
    method = method.substr(9);
  }

  std::string basis_name = cfg.basis_name.empty() ? "cc-pVDZ" : cfg.basis_name;
  BasisSet basis(basis_name, mol);
  auto integrals = std::make_shared<IntegralEngine>(mol, basis);
  const int nbf = basis.n_basis_functions();

  int n_elec = mol.n_electrons();
  int mult = mol.multiplicity();
  int na = (n_elec + mult - 1) / 2;
  int nb = n_elec - na;

  std::string ref = to_lower_copy(cfg.reference);
  if (ref.empty() || ref == "auto") {
    if (method == "rhf" || method == "rmp2" || method == "rmp3") {
      ref = "rhf";
    } else if (method == "rohf" || method == "romp2" || method == "romp3") {
      ref = "rohf";
    } else if (method == "uhf" || method == "ump2" || method == "ump3") {
      ref = "uhf";
    } else {
      ref = (mult == 1) ? "rhf" : "uhf";
    }
  }

  SCFConfig scf_conf;
  scf_conf.scf_type = "incore";
  scf_conf.eri_method = eri_mode;
  scf_conf.use_df = (eri_mode == "df");
  scf_conf.aux_basis_name = cfg.aux_basis_name.empty()
      ? (basis_name + "-RI")
      : cfg.aux_basis_name;
  scf_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;
  scf_conf.density_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;
  scf_conf.cholesky_threshold = std::min(1.0e-11, scf_conf.energy_threshold);
  scf_conf.df_threshold = std::min(1.0e-12, scf_conf.energy_threshold);
  scf_conf.print_level = 0;
  scf_conf.max_iterations = 200;

  if (guess_in && guess_in->has_scf && guess_in->scf_C_alpha.rows() == nbf) {
    scf_conf.has_initial_guess = true;
    scf_conf.init_C_alpha = guess_in->scf_C_alpha;
    if (guess_in->scf_C_beta.rows() == nbf) {
      scf_conf.init_C_beta = guess_in->scf_C_beta;
    }
  }

  SCFResult scf_res;
  if (ref == "rhf") {
    RHF rhf_solver(mol, basis, integrals, scf_conf);
    scf_res = rhf_solver.compute();
  } else if (ref == "rohf") {
    ROHF rohf_solver(mol, basis, integrals, na, nb, scf_conf);
    scf_res = rohf_solver.compute();
  } else {
    UHF uhf_solver(mol, basis, integrals, na, nb, scf_conf);
    scf_res = uhf_solver.compute();
  }

  if (guess_out) {
    guess_out->scf_C_alpha = scf_res.C_alpha;
    guess_out->scf_C_beta = scf_res.C_beta;
    guess_out->has_scf = (scf_res.C_alpha.rows() == nbf);
  }

  if (method == "scf" || method == "hf" || method == "rhf" || method == "uhf" || method == "rohf") {
    return scf_res.energy_total;
  }

  MP2Config mp2_conf;
  mp2_conf.scf_type = "incore";
  mp2_conf.eri_method = eri_mode;
  mp2_conf.use_df = (eri_mode == "df");
  mp2_conf.aux_basis_name = scf_conf.aux_basis_name;
  mp2_conf.energy_threshold = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;
  mp2_conf.df_threshold = std::min(1.0e-12, mp2_conf.energy_threshold);
  mp2_conf.cholesky_threshold = std::min(1.0e-11, mp2_conf.energy_threshold);
  mp2_conf.gradient_threshold = (cfg.orb_grad_thresh > 0.0) ? std::min(1.0e-10, cfg.orb_grad_thresh) : 1.0e-10;
  mp2_conf.opt_method = cfg.opt_method;
  mp2_conf.print_level = 0;
  mp2_conf.max_iterations = 100;

  if (guess_in && guess_in->has_omp && guess_in->omp_C_alpha.rows() == nbf) {
    mp2_conf.has_initial_guess = true;
    mp2_conf.init_C_alpha = guess_in->omp_C_alpha;
    if (guess_in->omp_C_beta.rows() == nbf) {
      mp2_conf.init_C_beta = guess_in->omp_C_beta;
    }
  }

  if (method == "mp2" || method == "rmp2" || method == "ump2" || method == "romp2") {
    if (method == "rmp2" || (method == "mp2" && ref == "rhf")) {
      foundation::RMP2 solver(mol, basis, integrals, scf_res, mp2_conf);
      return solver.compute().energy_total;
    } else if (method == "romp2" || (method == "mp2" && ref == "rohf")) {
      ROMP2 solver(mol, basis, integrals, scf_res, mp2_conf);
      return solver.compute().energy_total;
    } else {
      UMP2 solver(mol, basis, integrals, scf_res, mp2_conf);
      return solver.compute().energy_total;
    }
  }

  if (method == "mp3" || method == "rmp3" || method == "ump3" || method == "romp3") {
    if (method == "rmp3" || (method == "mp3" && ref == "rhf")) {
      foundation::RMP2 mp2_solver(mol, basis, integrals, scf_res, mp2_conf);
      MP2Result mp2_res = mp2_solver.compute();
      RMP3 mp3_solver(scf_res, mp2_res, mp2_conf, integrals);
      return mp3_solver.compute().e_total;
    } else if (method == "romp3" || (method == "mp3" && ref == "rohf")) {
      ROMP2 mp2_solver(mol, basis, integrals, scf_res, mp2_conf);
      MP2Result mp2_res = mp2_solver.compute();
      ROMP3 mp3_solver(scf_res, mp2_res, mp2_conf, integrals);
      return mp3_solver.compute().e_total;
    } else {
      UMP2 mp2_solver(mol, basis, integrals, scf_res, mp2_conf);
      MP2Result mp2_res = mp2_solver.compute();
      UMP3 mp3_solver(scf_res, mp2_res, mp2_conf, integrals);
      return mp3_solver.compute().e_total;
    }
  }

  if (method == "omp2") {
    OMP2 solver(mol, basis, integrals, scf_res, mp2_conf);
    MP2Result r = solver.compute();
    if (guess_out) {
      guess_out->omp_C_alpha = r.C_alpha;
      guess_out->omp_C_beta = r.C_beta;
      guess_out->has_omp = (r.C_alpha.rows() == nbf);
    }
    return r.energy_total;
  }

  if (method == "omp3") {
    OMP3 solver(mol, basis, integrals, scf_res, mp2_conf);
    MP3Result r = solver.compute_omp3();
    if (guess_out) {
      guess_out->omp_C_alpha = r.C_alpha;
      guess_out->omp_C_beta = r.C_beta;
      guess_out->has_omp = (r.C_alpha.rows() == nbf);
    }
    return r.e_total;
  }

  throw std::invalid_argument("Metode tidak dikenali pada HPCGeometryOptimizer: " + cfg.method);
}"""

    src = replace_function_body(
        src,
        r"GradCurvatureResult\s+compute_gradient_and_curvature_internal\s*\([^)]*\)\s*\{",
        new_gc_fn,
    )

    # 3d. Delegasikan evaluate_single_point_cpp ke evaluate_single_point_with_guess
    new_sp_wrapper = r"""double HPCGeometryOptimizer::evaluate_single_point_cpp(
    const Molecule& mol,
    const GeomOptConfig& cfg) {
  return evaluate_single_point_with_guess(mol, cfg, nullptr, nullptr);
}"""
    src = replace_function_body(
        src,
        r"double\s+HPCGeometryOptimizer::evaluate_single_point_cpp\s*\([^)]*\)\s*\{",
        new_sp_wrapper,
    )

    # 3e. Sambungkan opt_guess di optimize_molecule
    src = re.sub(
        r"(double\s+tr\s*=\s*cfg\.trust_radius_init;)",
        r"\1\n  WarmStartGuess opt_guess;",
        src,
        count=1,
    )
    src = re.sub(
        r"compute_gradient_and_curvature_internal\(cur_mol,\s*cfg\)",
        "compute_gradient_and_curvature_internal(cur_mol, cfg, &opt_guess)",
        src,
    )
    src = re.sub(
        r"evaluate_single_point_cpp\(make_displaced_molecule\(mol,\s*x_polish\),\s*cfg\)",
        "evaluate_single_point_with_guess(make_displaced_molecule(mol, x_polish), cfg, &opt_guess, nullptr)",
        src,
    )

    # 3f. Upgrade compute_frequencies: gunakan FD Gradien Analitik O(N) untuk DF + Warm-Start untuk stensil energi
    old_freq_hess = re.compile(
        r"double\s+e0\s*=\s*evaluate_single_point_cpp\(mol,\s*cfg\);\s*\n"
        r"[ \t]*Eigen::MatrixXd\s+hess\s*=\s*Eigen::MatrixXd::Zero\(ndof,\s*ndof\);.*?"
        r"hess\s*=\s*0\.5\s*\*\s*\(hess\s*\+\s*hess\.transpose\(\)\);",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not old_freq_hess.search(src):
        raise RuntimeError(f"Blok Hessian compute_frequencies tidak ditemukan di {p}")

    new_freq_hess = r"""WarmStartGuess freq_guess;
  double e0 = evaluate_single_point_with_guess(mol, cfg, nullptr, &freq_guess);
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(ndof, ndof);
  const double h = 2.0e-3;
  const int n_indep = 3 * std::max(0, natoms - 1);

  if (supports_analytic_df_gradient(cfg)) {
    // O(N) Hessian via Central Finite-Difference Gradien Analitik (hanya 6*(N-1) evaluasi!)
    GeomOptConfig grad_cfg = cfg;
    grad_cfg.project_trans_rot = false;
    const double inv_2h = 1.0 / (2.0 * h);
    for (int k = 0; k < n_indep; ++k) {
      Eigen::VectorXd xp = x0, xm = x0;
      xp(k) += h;
      xm(k) -= h;
      GradCurvatureResult gp = compute_analytic_df_gradient_internal(
          make_displaced_molecule(mol, xp), grad_cfg, &freq_guess, nullptr);
      GradCurvatureResult gm = compute_analytic_df_gradient_internal(
          make_displaced_molecule(mol, xm), grad_cfg, &freq_guess, nullptr);
      hess.col(k) = (gp.grad - gm.grad) * inv_2h;
    }
    // Lengkapi kolom atom terakhir dari invarian translasi eksak
    for (int xyz = 0; xyz < 3; ++xyz) {
      for (int r = 0; r < ndof; ++r) {
        double s = 0.0;
        for (int a = 0; a < natoms - 1; ++a) s += hess(r, 3 * a + xyz);
        hess(r, n_indep + xyz) = -s;
      }
    }
  } else {
    const double inv_h2 = 1.0 / (h * h);
    std::vector<double> ep_1d(n_indep, 0.0);
    std::vector<double> em_1d(n_indep, 0.0);

    for (int k = 0; k < n_indep; ++k) {
      Eigen::VectorXd xp = x0, xm = x0;
      xp(k) += h;
      xm(k) -= h;
      ep_1d[k] = evaluate_single_point_with_guess(make_displaced_molecule(mol, xp), cfg, &freq_guess, nullptr);
      em_1d[k] = evaluate_single_point_with_guess(make_displaced_molecule(mol, xm), cfg, &freq_guess, nullptr);
      hess(k, k) = (ep_1d[k] - 2.0 * e0 + em_1d[k]) * inv_h2;
    }

    for (int k = 0; k < n_indep; ++k) {
      for (int l = k + 1; l < n_indep; ++l) {
        Eigen::VectorXd xpp = x0, xmm = x0;
        xpp(k) += h; xpp(l) += h;
        xmm(k) -= h; xmm(l) -= h;
        const double epp = evaluate_single_point_with_guess(make_displaced_molecule(mol, xpp), cfg, &freq_guess, nullptr);
        const double emm = evaluate_single_point_with_guess(make_displaced_molecule(mol, xmm), cfg, &freq_guess, nullptr);
        const double h_kl = 0.5 * (epp + emm - ep_1d[k] - em_1d[k] - ep_1d[l] - em_1d[l] + 2.0 * e0) * inv_h2;
        hess(k, l) = h_kl;
        hess(l, k) = h_kl;
      }
    }

    for (int xyz = 0; xyz < 3; ++xyz) {
      for (int k = 0; k < n_indep; ++k) {
        double s = 0.0;
        for (int a = 0; a < natoms - 1; ++a) s += hess(k, 3 * a + xyz);
        hess(k, n_indep + xyz) = -s;
        hess(n_indep + xyz, k) = -s;
      }
    }
    for (int x1 = 0; x1 < 3; ++x1) {
      for (int x2 = 0; x2 < 3; ++x2) {
        double s = 0.0;
        for (int a = 0; a < natoms - 1; ++a) s += hess(n_indep + x1, 3 * a + x2);
        hess(n_indep + x1, n_indep + x2) = -s;
      }
    }
  }
  hess = 0.5 * (hess + hess.transpose());"""

    src = old_freq_hess.sub(new_freq_hess, src, count=1)

    # 3g. Warm-start berurutan di scan_bond_pes
    src = re.sub(
        r"std::vector<PESScanPoint>\s+pts;\s*\n"
        r"([ \t]*)for\s*\(double\s+r_ang\s*:\s*distances_angstrom\)\s*\{\s*\n"
        r"([ \t]*)double\s+r_bohr\s*=\s*r_ang\s*\*\s*ANGSTROM_TO_BOHR;\s*\n"
        r"([ \t]*)Eigen::VectorXd\s+x_scan\s*=\s*x0;\s*\n"
        r"([ \t]*)x_scan\.segment<3>\(3\s*\*\s*atom_j\)\s*=\s*x0\.segment<3>\(3\s*\*\s*atom_i\)\s*\+\s*r_bohr\s*\*\s*u_ij;\s*\n"
        r"([ \t]*)double\s+e_val\s*=\s*evaluate_single_point_cpp\(make_displaced_molecule\(mol,\s*x_scan\),\s*cfg\);",
        "std::vector<PESScanPoint> pts;\n"
        "  WarmStartGuess scan_guess;\n"
        r"\1for (double r_ang : distances_angstrom) {" + "\n"
        r"\2double r_bohr = r_ang * ANGSTROM_TO_BOHR;" + "\n"
        r"\3Eigen::VectorXd x_scan = x0;" + "\n"
        r"\4x_scan.segment<3>(3 * atom_j) = x0.segment<3>(3 * atom_i) + r_bohr * u_ij;" + "\n"
        r"\5double e_val = evaluate_single_point_with_guess(make_displaced_molecule(mol, x_scan), cfg, &scan_guess, &scan_guess);",
        src,
        count=1,
    )

    write_text(p, src, dry_run)


# =============================================================================
# 4. GUARD FIX: src/mp3/mp3.cc (UMP3 frozen-core guard)
# =============================================================================
def patch_mp3(root: Path, dry_run: bool) -> None:
    p = root / "src/mp3/mp3.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_UMP3_FROZEN_GUARD]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    target = re.compile(
        r"(const\s+int\s+na_act\s*=\s*no_a_\s*-\s*f_a;\s*\n"
        r"[ \t]*const\s+int\s+nb_act\s*=\s*no_b_\s*-\s*f_b;)",
        flags=re.MULTILINE,
    )
    if not target.search(src):
        raise RuntimeError(f"Anchor frozen-core UMP3 tidak ditemukan di {p}")

    replacement = (
        f"{marker}\n"
        r"\1"
        "\n    if (na_act <= 0 || nv_a_ <= 0) {\n"
        "      MP3Result empty_res;\n"
        "      empty_res.e_hf = scf_.energy_total;\n"
        "      empty_res.e_mp2 = mp2_.energy_mp2_corr;\n"
        "      empty_res.e_corr_total = empty_res.e_mp2;\n"
        "      empty_res.e_total = empty_res.e_hf + empty_res.e_corr_total;\n"
        "      return empty_res;\n"
        "    }"
    )
    src = target.sub(replacement, src, count=1)

    # Guard ukuran B_ia_P sebelum bottomRows
    src = re.sub(
        r"Bov_a_saved\s*=\s*std::move\(mp2_\.B_ia_P_alpha\);\s*\n"
        r"[ \t]*mp2_\.B_ia_P_alpha\s*=\s*Bov_a_saved\.bottomRows\(static_cast<Eigen::Index>\(na_act\)\s*\*\s*nv_a_\);",
        "if (mp2_.B_ia_P_alpha.rows() >= static_cast<Eigen::Index>(no_a_) * nv_a_) {\n"
        "        Bov_a_saved = std::move(mp2_.B_ia_P_alpha);\n"
        "        mp2_.B_ia_P_alpha = Bov_a_saved.bottomRows(static_cast<Eigen::Index>(na_act) * nv_a_);\n"
        "      }",
        src,
        count=1,
    )

    write_text(p, src, dry_run)


# =============================================================================
# 5. OPT: src/integrals/cholesky_eri.cc (single-pass packed HDF5 load)
# =============================================================================
def patch_cholesky_eri(root: Path, dry_run: bool) -> None:
    p = root / "src/integrals/cholesky_eri.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_CHOL_HDF5_LOAD]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    new_fn = r"""// [INJECTED_AUDIT_FIX_CHOL_HDF5_LOAD]
void CholeskyERI::load_from_hdf5(const std::string& filename, const std::string& dataset_name) {
  utils::HDF5TensorIO io(filename, utils::HDF5TensorIO::Mode::READ_ONLY);
  Eigen::Tensor<double, 4> tensor_in = io.read_tensor_4d(dataset_name);

  const long n_vec = tensor_in.dimension(0);
  n_basis_ = static_cast<int>(tensor_in.dimension(1));
  n_vectors_ = static_cast<int>(n_vec);
  const int n_tri = ao_tri_size(n_basis_);

  L_mat_.resize(n_tri, n_vectors_);
  #pragma omp parallel for schedule(static)
  for (int k = 0; k < n_vectors_; ++k) {
    double* dst = L_mat_.col(k).data();
    for (int r = 0; r < n_basis_; ++r) {
      const int base = r * (r + 1) / 2;
      for (int q = 0; q <= r; ++q) {
        dst[base + q] = tensor_in(k, r, q, 0);
      }
    }
  }
  decomposed_ = true;
}"""

    src = replace_function_body(
        src,
        r"void\s+CholeskyERI::load_from_hdf5\s*\([^)]*\)\s*\{",
        new_fn,
    )
    write_text(p, src, dry_run)


# =============================================================================
# 6. OPT: src/ints/integrals.cc (exact density-block screening in DF 2e grad)
# =============================================================================
def patch_integrals(root: Path, dry_run: bool) -> None:
    p = root / "src/ints/integrals.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_DF_2E_GRAD_SCREEN]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    # Tambahkan helper lambda screening blok densitas di compute_df_2e_gradient
    anchor = re.compile(
        r"(const\s+size_t\s+buf_2c_sz\s*=\s*std::max<size_t>\([^;]+\);)",
        flags=re.MULTILINE,
    )
    if not anchor.search(src):
        raise RuntimeError(f"Anchor buf_2c_sz tidak ditemukan di {p}")

    helper = (
        r"\1"
        f"\n\n  {marker}\n"
        "  auto block_has_density = [&](int bf1, int dim1, int bf2, int dim2, int bfR, int dimR) -> bool {\n"
        "    for (int p = 0; p < dimR; ++p) {\n"
        "      const double* g_col = Gamma_tilde_3c.col(bfR + p).data();\n"
        "      for (int j = 0; j < dim2; ++j) {\n"
        "        const int row_base = (bf2 + j) * n_prim_bf + bf1;\n"
        "        for (int i = 0; i < dim1; ++i) {\n"
        "          if (std::abs(g_col[row_base + i]) > 1e-14) return true;\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "    return false;\n"
        "  };"
    )
    src = anchor.sub(helper, src, count=1)

    # Pasang screening sebelum cint3c2e_ip1_sph dan cint3c2e_ip2_sph
    src = re.sub(
        r"(int\s+shls\[3\]\s*=\s*\{s1,\s*s2,\s*abs_r\};\s*\n[ \t]*if\s*\(!cint3c2e_ip1_sph)",
        "if (!block_has_density(bf1, dim1, bf2, dim2, bfR, dimR)) continue;\n          \\1",
        src,
        count=1,
    )
    src = re.sub(
        r"(int\s+shls\[3\]\s*=\s*\{s1,\s*s2,\s*abs_r\};\s*\n[ \t]*if\s*\(!cint3c2e_ip2_sph)",
        "if (!block_has_density(bf1, dim1, bf2, dim2, bfR, dimR)) continue;\n          \\1",
        src,
        count=1,
    )

    write_text(p, src, dry_run)


# =============================================================================
# 7. CLEANUP: src/scf/scf.cc (PetiteList schwarz fallback & skip FockBuilder in DF/CD)
# =============================================================================
def patch_scf(root: Path, dry_run: bool) -> None:
    p = root / "src/scf/scf.cc"
    src = read_text(p)
    marker = "// [INJECTED_AUDIT_FIX_SCF_CLEANUP]"
    if marker in src:
        print(f"[SKIP] {p} sudah dipatch.")
        return

    src = re.sub(
        r"if\s*\(pl_\s*&&\s*pl_->get_unique_quartets\(\)\.empty\(\)\)\s*pl_->build\(\);",
        f"{marker}\n  if (pl_ && pl_->get_unique_quartets().empty()) pl_->build(schwarz_, 1e-12);",
        src,
        count=1,
    )
    src = re.sub(
        r'if\s*\(config_\.scf_type\s*==\s*"direct"\)',
        'if (config_.scf_type == "direct" && !config_.use_df && config_.eri_method != "cholesky")',
        src,
    )
    write_text(p, src, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply mshqc audit bugfixes & remaining optimizations.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root direktori repositori mshqc")
    parser.add_argument("--dry-run", action="store_true", help="Simulasi tanpa menulis file")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"=== MSHQC Audit Fix & Optimization Injector ===\nRoot: {root}\n")

    patches = [
        ("molecule_sym.cc (Fix SEGFAULT)", patch_molecule_sym),
        ("omp_gradient.cc (Fix/Opt O(O^2 V) SOSCF/CPHF)", patch_omp_gradient),
        ("hpc_geom_opt.cc (Fix DF-ROHF + Warm-Start + O(N) Freq)", patch_hpc_geom_opt),
        ("mp3.cc (Guard UMP3 frozen-core)", patch_mp3),
        ("cholesky_eri.cc (Single-pass packed HDF5 load)", patch_cholesky_eri),
        ("integrals.cc (Density-block screening DF 2e grad)", patch_integrals),
        ("scf.cc (PetiteList schwarz & FockBuilder guard)", patch_scf),
    ]

    for label, fn in patches:
        print(f"-> {label}")
        fn(root, args.dry_run)

    print("\nSelesai. Rekomendasi verifikasi setelah build:")
    print("  1. Uji molekul bersimetri (H2O/C2v & O2/D2h) untuk memastikan molecule_sym.cc bebas segfault.")
    print("  2. Jalankan tests/verify_analytic_grad_2e.py termasuk kasus DF-ROHF (misal radikal OH/NH2).")
    print("  3. Uji compute_frequencies() DF-RHF/DF-OMP2 vs finite-difference energi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())