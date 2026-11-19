// Copyright 2026 Muhamad Syahrul Hidayat
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mshqc/gradient/hpc_geom_opt.h"
#include "mshqc/gradient/optimizer.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/scf/scf.h"
#include "mshqc/mp2/mp2.h"
#include "mshqc/mp3/mp3.h"
#include "mshqc/mp3/omp3.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {
namespace gradient {

namespace {

constexpr double HARTREE_TO_KCAL = 627.5094740631;
constexpr double HARTREE_TO_J = 4.3597447222071e-18;
constexpr double BOHR_TO_M = 5.29177210903e-11;
constexpr double AMU_TO_KG = 1.66053906660e-27;
constexpr double SPEED_OF_LIGHT_CMS = 2.99792458e10;
constexpr double PLANCK_SI = 6.62607015e-34;
constexpr double BOLTZMANN_SI = 1.380649e-23;
constexpr double PI = 3.14159265358979323846;

struct ScopedCoutSilencer {
  std::streambuf* old_buf = nullptr;
  std::ostringstream null_stream;
  explicit ScopedCoutSilencer(bool silence = true) {
    if (silence) {
      old_buf = std::cout.rdbuf(null_stream.rdbuf());
    }
  }
  ~ScopedCoutSilencer() {
    if (old_buf != nullptr) {
      std::cout.rdbuf(old_buf);
    }
  }
};

double get_exact_isotopic_mass(int Z) {
  static const std::map<int, double> m = {
    {1, 1.007825032},  {2, 4.002603254},  {3, 7.016003437},  {4, 9.01218307},
    {5, 11.00930536},  {6, 12.00000000},  {7, 14.003074004}, {8, 15.994914619},
    {9, 18.998403163}, {10, 19.99244018}, {11, 22.98976928}, {12, 23.98504170},
    {13, 26.98153853}, {14, 27.976926535},{15, 30.973761998},{16, 31.972071174},
    {17, 34.96885268}, {18, 39.96238312}, {35, 78.918338},   {53, 126.904472}
  };
  auto it = m.find(Z);
  return (it != m.end()) ? it->second : static_cast<double>(2 * Z);
}

std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

Molecule make_displaced_molecule(const Molecule& base_mol, const Eigen::VectorXd& coords_bohr) {
  Molecule m(base_mol.charge(), base_mol.multiplicity());
  const size_t natoms = base_mol.n_atoms();
  for (size_t i = 0; i < natoms; ++i) {
    m.add_atom(
        base_mol.atom(i).atomic_number,
        coords_bohr(3 * i + 0),
        coords_bohr(3 * i + 1),
        coords_bohr(3 * i + 2));
  }
  return m;
}

// Membangun basis ortonormal subruang vibrasi internal V_int (3N x N_vib)
// yang bebas dari 5 atau 6 mode translasi dan rotasi kaku (Proyeksi Eckart).

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

Eigen::MatrixXd build_internal_vibrational_subspace(
    const Eigen::VectorXd& coords_bohr,
    const std::vector<double>& masses_amu,
    bool mass_weighted = false) {

  const int natoms = static_cast<int>(masses_amu.size());
  const int ndof = 3 * natoms;
  if (natoms <= 1 || coords_bohr.size() != ndof) {
    return Eigen::MatrixXd::Zero(ndof, 0);
  }

  double total_mass = 0.0;
  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  for (int i = 0; i < natoms; ++i) {
    double m = (masses_amu[i] > 0.0) ? masses_amu[i] : 1.0;
    total_mass += m;
    com += m * coords_bohr.segment<3>(3 * i);
  }
  com /= total_mass;

  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ndof, 6);
  for (int i = 0; i < natoms; ++i) {
    double w = mass_weighted ? std::sqrt((masses_amu[i] > 0.0) ? masses_amu[i] : 1.0) : 1.0;
    Eigen::Vector3d r = coords_bohr.segment<3>(3 * i) - com;
    D(3 * i + 0, 0) = w;
    D(3 * i + 1, 1) = w;
    D(3 * i + 2, 2) = w;

    D(3 * i + 1, 3) = -w * r(2);
    D(3 * i + 2, 3) =  w * r(1);
    D(3 * i + 0, 4) =  w * r(2);
    D(3 * i + 2, 4) = -w * r(0);
    D(3 * i + 0, 5) = -w * r(1);
    D(3 * i + 1, 5) =  w * r(0);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullU);
  const Eigen::MatrixXd& U = svd.matrixU();
  const Eigen::VectorXd& s = svd.singularValues();

  int n_tr = 0;
  for (int k = 0; k < s.size(); ++k) {
    if (s(k) > 1.0e-6) ++n_tr;
  }
  const int n_vib = std::max(0, ndof - n_tr);
  Eigen::MatrixXd V_int = U.rightCols(n_vib);
  for (int c = 0; c < V_int.cols(); ++c) {
    int max_idx = 0;
    V_int.col(c).cwiseAbs().maxCoeff(&max_idx);
    if (V_int(max_idx, c) < 0.0) {
      V_int.col(c) = -V_int.col(c);
    }
  }
  return V_int;
}

// Membangun matriks Hessian awal model Lindh-Schlegel (ikatan + sudut valensi)
Eigen::MatrixXd build_lindh_model_hessian(const Eigen::VectorXd& x, int natoms) {
  const int ndof = 3 * natoms;
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(ndof, ndof) * 0.05;

  // 1. Suku Regangan Ikatan (Bond Stretch)
  for (int i = 0; i < natoms; ++i) {
    for (int j = i + 1; j < natoms; ++j) {
      Eigen::Vector3d rij_vec = x.segment<3>(3 * i) - x.segment<3>(3 * j);
      double rij = rij_vec.norm();
      if (rij > 1.0e-3 && rij < 5.5) {
        Eigen::Vector3d u_ij = rij_vec / rij;
        double k_stretch = 0.45 * std::exp(-0.45 * (rij - 1.8));
        Eigen::Matrix3d P_ij = k_stretch * (u_ij * u_ij.transpose());
        H.block<3, 3>(3 * i, 3 * i) += P_ij;
        H.block<3, 3>(3 * j, 3 * j) += P_ij;
        H.block<3, 3>(3 * i, 3 * j) -= P_ij;
        H.block<3, 3>(3 * j, 3 * i) -= P_ij;
      }
    }
  }

  // 2. Suku Tekukan Sudut Valensi (Valence Angle Bend) untuk N >= 3
  for (int j = 0; j < natoms; ++j) {
    for (int i = 0; i < natoms; ++i) {
      if (i == j) continue;
      Eigen::Vector3d u_ji = x.segment<3>(3 * i) - x.segment<3>(3 * j);
      double r_ji = u_ji.norm();
      if (r_ji < 1.0e-3 || r_ji > 4.5) continue;
      u_ji /= r_ji;

      for (int k = i + 1; k < natoms; ++k) {
        if (k == j) continue;
        Eigen::Vector3d u_jk = x.segment<3>(3 * k) - x.segment<3>(3 * j);
        double r_jk = u_jk.norm();
        if (r_jk < 1.0e-3 || r_jk > 4.5) continue;
        u_jk /= r_jk;

        double cos_theta = std::clamp(u_ji.dot(u_jk), -0.999999, 0.999999);
        double sin_theta = std::sqrt(std::max(1.0e-8, 1.0 - cos_theta * cos_theta));
        if (sin_theta < 1.0e-3) continue;

        Eigen::Vector3d b_i = (cos_theta * u_ji - u_jk) / (r_ji * sin_theta);
        Eigen::Vector3d b_k = (cos_theta * u_jk - u_ji) / (r_jk * sin_theta);
        Eigen::Vector3d b_j = -(b_i + b_k);

        Eigen::VectorXd b_vec = Eigen::VectorXd::Zero(ndof);
        b_vec.segment<3>(3 * i) = b_i;
        b_vec.segment<3>(3 * j) = b_j;
        b_vec.segment<3>(3 * k) = b_k;

        double k_bend = 0.16 * std::exp(-0.4 * ((r_ji - 1.8) + (r_jk - 1.8)));
        H.noalias() += k_bend * (b_vec * b_vec.transpose());
      }
    }
  }

  return 0.5 * (H + H.transpose());
}

struct GradCurvatureResult {
  double energy = 0.0;
  Eigen::VectorXd grad;
  Eigen::VectorXd diag_curv;
  Eigen::VectorXi has_curv;
};

// [INJECTED_PRIORITAS_2E_ANALYTIC_GRAD]
Eigen::MatrixXd compute_nuclear_repulsion_gradient(const Molecule& mol) {
  const int natoms = static_cast<int>(mol.n_atoms());
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(natoms, 3);
  for (int a = 0; a < natoms; ++a) {
    const double Za = static_cast<double>(mol.atom(a).atomic_number);
    const Eigen::Vector3d ra(mol.atom(a).x, mol.atom(a).y, mol.atom(a).z);
    for (int b = a + 1; b < natoms; ++b) {
      const double Zb = static_cast<double>(mol.atom(b).atomic_number);
      const Eigen::Vector3d rb(mol.atom(b).x, mol.atom(b).y, mol.atom(b).z);
      const Eigen::Vector3d rab = ra - rb;
      const double r2 = rab.squaredNorm();
      const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
      const Eigen::Vector3d f = -Za * Zb * inv_r3 * rab;
      g.row(a) += f.transpose();
      g.row(b) -= f.transpose();
    }
  }
  return g;
}

Eigen::MatrixXd compute_df_metric_inv_sqrt(
    const BasisSet& primary_basis,
    const BasisSet& aux_basis,
    const std::shared_ptr<IntegralEngine>& df_ints,
    double cutoff) {

  const int n_shells_prim = primary_basis.n_shells();
  const int n_shells_aux  = aux_basis.n_shells();
  const int n_aux = aux_basis.n_basis_functions();

  std::vector<int> aux_starts(n_shells_aux), aux_sizes(n_shells_aux);
  int off = 0, max_a = 1;
  for (int i = 0; i < n_shells_aux; ++i) {
    aux_starts[i] = off;
    aux_sizes[i] = aux_basis.shell(i).n_functions();
    max_a = std::max(max_a, aux_sizes[i]);
    off += aux_sizes[i];
  }

  Eigen::MatrixXd J_metric = Eigen::MatrixXd::Zero(n_aux, n_aux);
  const size_t buf_sz = std::max<size_t>(64, static_cast<size_t>(max_a) * max_a);
  #pragma omp parallel
  {
    std::vector<double> buf(buf_sz);
    #pragma omp for schedule(dynamic, 1) nowait
    for (int R = 0; R < n_shells_aux; ++R) {
      const int abs_R = n_shells_prim + R, dim_R = aux_sizes[R], st_R = aux_starts[R];
      for (int S = 0; S <= R; ++S) {
        const int abs_S = n_shells_prim + S, dim_S = aux_sizes[S], st_S = aux_starts[S];
        if (!df_ints->compute_2c2e_block_raw(abs_R, abs_S, buf.data())) continue;
        for (int s = 0; s < dim_S; ++s) {
          for (int r = 0; r < dim_R; ++r) {
            double val = buf[r + dim_R * s];
            J_metric(st_R + r, st_S + s) = val;
            if (st_R + r != st_S + s) J_metric(st_S + s, st_R + r) = val;
          }
        }
      }
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J_metric);
  const Eigen::VectorXd& evals = es.eigenvalues();
  const Eigen::MatrixXd& evecs = es.eigenvectors();
  double max_eval = (evals.size() > 0) ? std::max(1.0, evals.maxCoeff()) : 1.0;
  double rel_cond = (cutoff > 0.0 && cutoff <= 1e-10) ? cutoff : 1e-12;
  double eig_tol = std::max(1e-14, rel_cond * max_eval);

  int rank = 0;
  for (int i = 0; i < n_aux; ++i) if (evals(i) >= eig_tol) ++rank;
  Eigen::MatrixXd U_kept(n_aux, rank);
  Eigen::VectorXd inv_sq(rank);
  int c = 0;
  for (int i = 0; i < n_aux; ++i) {
    if (evals(i) >= eig_tol) {
      U_kept.col(c) = evecs.col(i);
      inv_sq(c) = 1.0 / std::sqrt(evals(i));
      ++c;
    }
  }
  Eigen::MatrixXd U_sc = U_kept * inv_sq.asDiagonal();
  Eigen::MatrixXd J_inv_sqrt(n_aux, n_aux);
  constexpr int tile = 64;
  #pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < n_aux; r += tile) {
    int rows = std::min(tile, n_aux - r);
    J_inv_sqrt.middleRows(r, rows).noalias() = U_sc.middleRows(r, rows) * U_kept.transpose();
  }
  return J_inv_sqrt;
}

bool supports_analytic_df_gradient(const GeomOptConfig& cfg) {
  if (!cfg.use_analytic_gradient) return false;
  std::string method = to_lower_copy(cfg.method);
  std::string eri    = to_lower_copy(cfg.eri_method);
  if (eri == "ri") eri = "df";
  if (method.rfind("df-", 0) == 0 || method.rfind("df_", 0) == 0) {
    eri = "df";
    method = method.substr(3);
  }
  if (eri != "df") return false;
  return (method == "scf" || method == "hf" || method == "rhf" || method == "uhf" || method == "omp2");
}

GradCurvatureResult compute_analytic_df_gradient_internal(
    const Molecule& mol,
    const GeomOptConfig& cfg) {

  ScopedCoutSilencer silencer(true);
  const int natoms = static_cast<int>(mol.n_atoms());
  const int ndof   = 3 * natoms;

  GradCurvatureResult out;
  out.grad      = Eigen::VectorXd::Zero(ndof);
  out.diag_curv = Eigen::VectorXd::Zero(ndof);
  out.has_curv  = Eigen::VectorXi::Zero(ndof);

  std::string method = to_lower_copy(cfg.method);
  if (method.rfind("df-", 0) == 0 || method.rfind("df_", 0) == 0) method = method.substr(3);

  const std::string basis_name = cfg.basis_name.empty() ? "cc-pVDZ" : cfg.basis_name;
  const std::string aux_name   = cfg.aux_basis_name.empty() ? (basis_name + "-RI") : cfg.aux_basis_name;

  BasisSet basis(basis_name, mol);
  auto integrals = std::make_shared<IntegralEngine>(mol, basis);
  const int nbf = basis.n_basis_functions();

  const int n_elec = mol.n_electrons();
  const int mult   = mol.multiplicity();
  const int na     = (n_elec + mult - 1) / 2;
  const int nb     = n_elec - na;

  std::string ref = to_lower_copy(cfg.reference);
  if (ref.empty() || ref == "auto") {
    if (method == "rhf") ref = "rhf";
    else if (method == "uhf") ref = "uhf";
    else ref = (mult == 1) ? "rhf" : "uhf";
  }

  SCFConfig scf_conf;
  scf_conf.scf_type          = "incore";
  scf_conf.eri_method        = "df";
  scf_conf.use_df            = true;
  scf_conf.aux_basis_name    = aux_name;
  scf_conf.energy_threshold  = (cfg.energy_thresh > 0.0) ? std::min(1.0e-12, cfg.energy_thresh) : 1.0e-12;
  scf_conf.density_threshold = scf_conf.energy_threshold;
  scf_conf.df_threshold      = std::min(1.0e-12, scf_conf.energy_threshold);
  scf_conf.print_level       = 0;
  scf_conf.max_iterations    = 200;

  SCFResult scf_res;
  if (ref == "rhf") {
    RHF solver(mol, basis, integrals, scf_conf);
    scf_res = solver.compute();
  } else {
    UHF solver(mol, basis, integrals, na, nb, scf_conf);
    scf_res = solver.compute();
  }

  Eigen::MatrixXd P_eff_ao, W_eff_ao, Gamma_3c, gamma_2c;
  const int n_aux = static_cast<int>(scf_res.L_mat.cols());

  if (method == "omp2") {
    MP2Config mp2_conf;
    mp2_conf.scf_type           = "incore";
    mp2_conf.eri_method         = "df";
    mp2_conf.use_df             = true;
    mp2_conf.aux_basis_name     = aux_name;
    mp2_conf.energy_threshold   = scf_conf.energy_threshold;
    mp2_conf.df_threshold       = scf_conf.df_threshold;
    mp2_conf.gradient_threshold = (cfg.orb_grad_thresh > 0.0) ? std::min(1.0e-10, cfg.orb_grad_thresh) : 1.0e-10;
    mp2_conf.opt_method         = cfg.opt_method;
    mp2_conf.print_level        = 0;
    mp2_conf.max_iterations     = 100;

    OMP2 omp2_solver(mol, basis, integrals, scf_res, mp2_conf);
    MP2Result omp2_res = omp2_solver.compute();
    out.energy = omp2_res.energy_total;

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
    W_eff_ao = Ca_o * scf_res.orbital_energies_alpha.head(na).asDiagonal() * Ca_o.transpose();
    if (is_rhf) {
      W_eff_ao *= 2.0;
    } else if (nb > 0) {
      const Eigen::MatrixXd Cb_o = scf_res.C_beta.leftCols(nb);
      W_eff_ao.noalias() += Cb_o * scf_res.orbital_energies_beta.head(nb).asDiagonal() * Cb_o.transpose();
    }

    Eigen::Map<const Eigen::VectorXd> P_flat(P_eff_ao.data(), nbf * nbf);
    Eigen::VectorXd c_vec = scf_res.L_mat.transpose() * P_flat;
    Gamma_3c.resize(static_cast<Eigen::Index>(nbf) * nbf, n_aux);

    const Eigen::MatrixXd Cb_o = (!is_rhf && nb > 0) ? scf_res.C_beta.leftCols(nb) : Eigen::MatrixXd();
    #pragma omp parallel
    {
      Eigen::MatrixXd G_P(nbf, nbf), T_a(nbf, na), T_b;
      if (!is_rhf && nb > 0) T_b.resize(nbf, nb);

      #pragma omp for schedule(static) nowait
      for (int P = 0; P < n_aux; ++P) {
        Eigen::Map<const Eigen::MatrixXd> B_P(scf_res.L_mat.col(P).data(), nbf, nbf);
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

    gamma_2c.resize(n_aux, n_aux);
    constexpr int tile_g = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < n_aux; r += tile_g) {
      int rows = std::min(tile_g, n_aux - r);
      gamma_2c.middleRows(r, rows).noalias() = Gamma_3c.middleCols(r, rows).transpose() * scf_res.L_mat;
    }
    gamma_2c = 0.5 * (gamma_2c + gamma_2c.transpose());
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
}

// Mengevaluasi energi, gradien nuklir Central Finite-Difference, sekaligus
// mengekstrak kurvatur turunan kedua lokal d^2E/dx_i^2 = (E+ - 2*E0 + E-) / h^2
// dari titik-titik stensil yang sama tanpa tambahan evaluasi single-point.
GradCurvatureResult compute_gradient_and_curvature_internal(
    const Molecule& mol,
    const GeomOptConfig& cfg) {

  if (supports_analytic_df_gradient(cfg)) {
    return compute_analytic_df_gradient_internal(mol, cfg);
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

  GradCurvatureResult out;
  out.energy = HPCGeometryOptimizer::evaluate_single_point_cpp(mol, cfg);
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

      const double ep = HPCGeometryOptimizer::evaluate_single_point_cpp(
          make_displaced_molecule(mol, xp), cfg);
      const double em = HPCGeometryOptimizer::evaluate_single_point_cpp(
          make_displaced_molecule(mol, xm), cfg);

      out.grad(3 * a + xyz) = (ep - em) / (2.0 * h);
      const double curv = (ep - 2.0 * out.energy + em) / (h * h);
      if (std::isfinite(curv) && curv > 1.0e-2 && curv < 20.0) {
        out.diag_curv(3 * a + xyz) = curv;
        out.has_curv(3 * a + xyz) = 1;
      }
    }
  }

  // Suku atom terakhir dari invarian translasi eksak (Hukum III Newton)
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

} // namespace

double HPCGeometryOptimizer::evaluate_single_point_cpp(
    const Molecule& mol,
    const GeomOptConfig& cfg) {

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
    return solver.compute().energy_total;
  }

  if (method == "omp3") {
    OMP3 solver(mol, basis, integrals, scf_res, mp2_conf);
    return solver.compute_omp3().e_total;
  }

  throw std::invalid_argument("Metode tidak dikenali pada HPCGeometryOptimizer: " + cfg.method);
}

Eigen::VectorXd HPCGeometryOptimizer::project_out_trans_rot(
    const Eigen::VectorXd& coords_bohr,
    const Eigen::VectorXd& grad,
    const std::vector<double>& masses_amu) {

  const int natoms = static_cast<int>(masses_amu.size());
  const int ndof = 3 * natoms;
  if (natoms <= 1 || coords_bohr.size() != ndof || grad.size() != ndof) {
    return Eigen::VectorXd::Zero(ndof);
  }

  Eigen::MatrixXd V_int = build_internal_vibrational_subspace(coords_bohr, masses_amu, false);
  if (V_int.cols() == 0) {
    return Eigen::VectorXd::Zero(ndof);
  }
  Eigen::VectorXd projected = V_int * (V_int.transpose() * grad);
  symmetrize_cartesian_vector_sync(coords_bohr, projected);
  return projected;
}

std::pair<double, Eigen::VectorXd> HPCGeometryOptimizer::compute_nuclear_gradient_cpp(
    const Molecule& mol,
    const GeomOptConfig& cfg) {
  GradCurvatureResult res = compute_gradient_and_curvature_internal(mol, cfg);
  return {res.energy, res.grad};
}

GeomOptResult HPCGeometryOptimizer::optimize_molecule(
    Molecule& mol,
    const GeomOptConfig& cfg) {

  auto t_start = std::chrono::high_resolution_clock::now();
  const int natoms = static_cast<int>(mol.n_atoms());
  const int ndof = 3 * natoms;

  Eigen::VectorXd x(ndof);
  std::vector<double> masses(natoms);
  std::vector<std::string> symbols(natoms);

  for (int i = 0; i < natoms; ++i) {
    int z = mol.atom(i).atomic_number;
    x(3 * i + 0) = mol.atom(i).x;
    x(3 * i + 1) = mol.atom(i).y;
    x(3 * i + 2) = mol.atom(i).z;
    masses[i] = get_exact_isotopic_mass(z);
    symbols[i] = get_element_symbol(z);
  }

  Eigen::MatrixXd H = build_lindh_model_hessian(x, natoms);

  TrustRegionConfig tr_cfg;
  tr_cfg.max_micro_iter = std::max(30, 2 * ndof);
  tr_cfg.micro_thresh = 1.0e-10;
  tr_cfg.precond_shift = 1.0e-3;
  TrustRegionSOSCF tr_soscf(tr_cfg);

  if (!cfg.xyz_traj_file.empty()) {
    std::ofstream ofs(cfg.xyz_traj_file, std::ios::trunc);
  }

  GeomOptResult res;
  res.method = cfg.method;
  res.eri_method = cfg.eri_method;
  res.basis_name = cfg.basis_name;

  Eigen::VectorXd x_prev = x;
  Eigen::VectorXd g_prev = Eigen::VectorXd::Zero(ndof);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(ndof);
  double e_prev = 0.0;
  double pred_de_last = 0.0;
  bool hit_boundary_last = false;
  double tr = cfg.trust_radius_init;

  if (cfg.verbose) {
    std::cout << "============================================================================\n";
    std::cout << " MSHQC TRUST-REGION SOSCF GEOMETRY OPTIMIZER | Method: " << cfg.method
              << " (" << cfg.eri_method << ") | Basis: " << cfg.basis_name << "\n";
    std::cout << "----------------------------------------------------------------------------\n";
    std::cout << " Iter |   Total Energy (Ha)   |    dE (Ha)   |  RMS Grad  |  MAX Grad  |  Step RMS\n";
    std::cout << "----------------------------------------------------------------------------\n";
  }

  for (int iter = 0; iter < cfg.max_iter; ++iter) {
    Molecule cur_mol = make_displaced_molecule(mol, x);
    GradCurvatureResult gc = compute_gradient_and_curvature_internal(cur_mol, cfg);
    double energy = gc.energy;
    g = gc.grad;

    // Step Rejection / Backtracking Trust-Region jika langkah menaikkan energi
    if (iter > 0 && energy > e_prev + 1.0e-7) {
      tr = std::max(cfg.trust_radius_min, 0.4 * (x - x_prev).norm());
      x = 0.5 * (x + x_prev);
      cur_mol = make_displaced_molecule(mol, x);
      gc = compute_gradient_and_curvature_internal(cur_mol, cfg);
      energy = gc.energy;
      g = gc.grad;
    }

    // Injeksi kurvatur lokal eksak dari stensil Central Finite-Difference:
    // Untuk molekul diatomik (N=2), H_{00} = H_{11} = -H_{01} = d^2E/dz_0^2 adalah eksak!
    if (natoms == 2) {
      for (int xyz = 0; xyz < 3; ++xyz) {
        if (gc.has_curv(xyz)) {
          const double k_exact = gc.diag_curv(xyz);
          H(xyz, xyz) = k_exact;
          H(3 + xyz, 3 + xyz) = k_exact;
          H(xyz, 3 + xyz) = -k_exact;
          H(3 + xyz, xyz) = -k_exact;
        }
      }
    } else if (iter == 0) {
      // Pada poliatomik (N >= 3), jangkarkan diagonal Hessian awal dengan kurvatur FD lokal
      for (int a = 0; a < natoms - 1; ++a) {
        for (int xyz = 0; xyz < 3; ++xyz) {
          int idx = 3 * a + xyz;
          if (gc.has_curv(idx)) {
            H(idx, idx) = 0.7 * gc.diag_curv(idx) + 0.3 * H(idx, idx);
          }
        }
      }
    }

    double g_rms = std::sqrt(g.squaredNorm() / static_cast<double>(ndof));
    double g_max = g.cwiseAbs().maxCoeff();
    double de = (iter == 0) ? 0.0 : (energy - e_prev);

    res.energy_history.push_back(energy);
    res.g_rms_history.push_back(g_rms);

    if (!cfg.xyz_traj_file.empty()) {
      std::ofstream ofs(cfg.xyz_traj_file, std::ios::app);
      ofs << natoms << "\n";
      ofs << std::fixed << std::setprecision(10)
          << "Step " << iter << " E=" << energy << " Ha g_rms=" << g_rms << "\n";
      for (int a = 0; a < natoms; ++a) {
        ofs << std::left << std::setw(4) << symbols[a]
            << std::right << std::setw(16) << x(3 * a + 0) * BOHR_TO_ANGSTROM
            << std::setw(16) << x(3 * a + 1) * BOHR_TO_ANGSTROM
            << std::setw(16) << x(3 * a + 2) * BOHR_TO_ANGSTROM << "\n";
      }
    }

    // Update Hessian Bofill (kombinasi konveks SR1 + PSB) untuk poliatomik (N >= 3)
    // dan adaptasi radius Trust-Region Steihaug-Toint
    if (iter > 0) {
      Eigen::VectorXd s = x - x_prev;
      Eigen::VectorXd y = g - g_prev;
      const double s_norm = s.norm();
      const double y_norm = y.norm();

      const bool allow_bofill = (natoms >= 3) || (natoms == 2 && !gc.has_curv.any());
      if (allow_bofill && s_norm > 1.0e-4 && y_norm > 1.0e-6) {
        Eigen::VectorXd r = y - H * s;
        const double s2 = s.squaredNorm();
        const double r2 = r.squaredNorm();
        const double sr = s.dot(r);

        if (s2 > 1.0e-12 && r2 > 1.0e-16) {
          double phi = (sr * sr) / (r2 * s2);
          phi = std::clamp(phi, 0.0, 1.0);

          Eigen::MatrixXd dH_psb =
              (r * s.transpose() + s * r.transpose()) / s2 -
              (sr / (s2 * s2)) * (s * s.transpose());

          Eigen::MatrixXd dH = (1.0 - phi) * dH_psb;
          if (std::abs(sr) > 1.0e-8 * std::sqrt(s2 * r2)) {
            dH.noalias() += phi * ((r * r.transpose()) / sr);
          }
          H += dH;
          H = 0.5 * (H + H.transpose());
        }
      }

      // TIMPA DENGAN BLOK INI:
      if (std::abs(pred_de_last) > 1.0e-7) {
        const double rho = de / pred_de_last;
        if (rho < 0.25) {
          tr = std::max(cfg.trust_radius_min, 0.5 * s_norm);
        } else if (rho > 0.75 && hit_boundary_last) {
          tr = std::min(cfg.trust_radius_max, 1.5 * tr);
        }
      }
    }

    // Proyeksikan ke subruang vibrasi internal bebas translasi-rotasi (Eckart)
    // lalu pecahkan langkah kuadratik menggunakan TrustRegionSOSCF (Steihaug-Toint TCG)
    Eigen::MatrixXd V_int = build_internal_vibrational_subspace(x, masses, false);
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndof);

    if (V_int.cols() > 0) {
      Eigen::VectorXd g_int = V_int.transpose() * g;
      Eigen::MatrixXd H_int = V_int.transpose() * H * V_int;
      H_int = 0.5 * (H_int + H_int.transpose());

      // Regulasi spektral ringan agar preconditioner dan kelengkungan terjaga positif-definit
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_int);
      Eigen::VectorXd evals = es.eigenvalues();
      Eigen::MatrixXd evecs = es.eigenvectors();
      for (int k = 0; k < evals.size(); ++k) {
        evals(k) = std::clamp(evals(k), 2.0e-2, 20.0);
      }
      H_int = evecs * evals.asDiagonal() * evecs.transpose();
      Eigen::VectorXd diag_hess = H_int.diagonal().cwiseMax(2.0e-2);

      TrustRegionResult tr_res = tr_soscf.solve(
          g_int,
          diag_hess,
          tr,
          [&H_int](const Eigen::VectorXd& p) -> Eigen::VectorXd {
            return H_int * p;
          });

      dx = V_int * tr_res.step;
      pred_de_last = tr_res.predicted_energy_change;
      hit_boundary_last = tr_res.hit_boundary;
    }

    if (cfg.project_trans_rot) {
      dx = project_out_trans_rot(x, dx, masses);
    }

    double dx_rms = std::sqrt(dx.squaredNorm() / static_cast<double>(ndof));
    double dx_max = dx.cwiseAbs().maxCoeff();

    if (cfg.verbose) {
      std::cout << " " << std::setw(4) << iter
                << " | " << std::fixed << std::setprecision(10) << std::setw(18) << energy
                << " | " << std::scientific << std::setprecision(3) << std::setw(12) << de
                << " | " << std::setw(10) << g_rms
                << " | " << std::setw(10) << g_max
                << " | " << std::setw(10) << dx_rms << std::fixed << "\n";
    }

    // 2. Di optimize_molecule (baris 563-575):
    const double eff_g_rms_tol = (cfg.g_rms_tol > 0.0) ? cfg.g_rms_tol : 1.0e-8;
    const double eff_g_max_tol = (cfg.g_max_tol > 0.0) ? cfg.g_max_tol : 3.0e-8;
    const double eff_dx_rms_tol = (cfg.dx_rms_tol > 0.0) ? cfg.dx_rms_tol : 3.0e-7;
    const double eff_dx_max_tol = (cfg.dx_max_tol > 0.0) ? cfg.dx_max_tol : 1.0e-6;

    bool grad_ok = (g_rms < eff_g_rms_tol) && (g_max < eff_g_max_tol);
    bool step_ok = (dx_rms < eff_dx_rms_tol) && (dx_max < eff_dx_max_tol);
    bool tight_grad = (g_rms < 5.0e-9) && (g_max < 1.5e-8);
    bool quadratic_polish_ok = (iter >= 3) && (std::abs(de) < 1.0e-8) &&
                               (dx_rms < 5.0e-7) && (g_rms < std::min(eff_g_rms_tol, 1.0e-7));
    bool stationary_ok = (iter >= 4) && (std::abs(de) < 1.0e-9) &&
                         (dx_rms < 5.0e-7) && (g_rms < std::min(eff_g_rms_tol, 1.0e-7));

    if ((grad_ok && step_ok) || tight_grad || quadratic_polish_ok || stationary_ok || natoms <= 1) {
      if (natoms > 1 && pred_de_last <= 0.0 && dx_rms > 0.0 && dx_rms < 1.0e-4) {
        Eigen::VectorXd x_polish = x + dx;
        double e_polish = evaluate_single_point_cpp(make_displaced_molecule(mol, x_polish), cfg);
        if (e_polish <= energy + 1.0e-12) {
            x = x_polish;
            if (!res.energy_history.empty()) {
                res.energy_history.back() = e_polish;
            }
        }
      }
      res.converged = true;
      break;
    }

    x_prev = x;
    g_prev = g;
    e_prev = energy;
    x += dx;
  }

  res.coordinates_bohr.resize(natoms, 3);
  res.coordinates_angstrom.resize(natoms, 3);
  res.gradient_ha_bohr.resize(natoms, 3);

  for (int i = 0; i < natoms; ++i) {
    double xb = x(3 * i + 0);
    double yb = x(3 * i + 1);
    double zb = x(3 * i + 2);
    mol.set_atom_position(i, xb, yb, zb);

    res.coordinates_bohr(i, 0) = xb;
    res.coordinates_bohr(i, 1) = yb;
    res.coordinates_bohr(i, 2) = zb;

    res.coordinates_angstrom(i, 0) = xb * BOHR_TO_ANGSTROM;
    res.coordinates_angstrom(i, 1) = yb * BOHR_TO_ANGSTROM;
    res.coordinates_angstrom(i, 2) = zb * BOHR_TO_ANGSTROM;

    res.gradient_ha_bohr(i, 0) = g(3 * i + 0);
    res.gradient_ha_bohr(i, 1) = g(3 * i + 1);
    res.gradient_ha_bohr(i, 2) = g(3 * i + 2);
  }

  for (int i = 0; i < natoms; ++i) {
    for (int j = i + 1; j < natoms; ++j) {
      double dist_a = (res.coordinates_angstrom.row(i) - res.coordinates_angstrom.row(j)).norm();
      if (dist_a < 3.0 || natoms <= 4) {
        std::string key = symbols[i] + std::to_string(i + 1) + "-" + symbols[j] + std::to_string(j + 1);
        res.bond_lengths_angstrom[key] = dist_a;
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  res.wall_time_sec = std::chrono::duration<double>(t_end - t_start).count();
  res.iterations = static_cast<int>(res.energy_history.size());
  res.initial_energy = res.energy_history.empty() ? 0.0 : res.energy_history.front();
  res.final_energy = res.energy_history.empty() ? 0.0 : res.energy_history.back();
  res.final_g_rms = res.g_rms_history.empty() ? 0.0 : res.g_rms_history.back();
  res.final_g_max = g.cwiseAbs().maxCoeff();

  if (cfg.verbose) {
    std::cout << "============================================================================\n";
    std::cout << " Status             : " << (res.converged ? "CONVERGED" : "MAX_ITER REACHED")
              << " in " << res.iterations << " steps (" << std::fixed << std::setprecision(2)
              << res.wall_time_sec << " s)\n";
    std::cout << " Initial Energy     : " << std::setprecision(10) << res.initial_energy << " Ha\n";
    std::cout << " Final Opt Energy   : " << std::setprecision(10) << res.final_energy << " Ha\n";
    std::cout << " Relaxation Lowering: " << std::setprecision(4)
              << (res.final_energy - res.initial_energy) * HARTREE_TO_KCAL << " kcal/mol\n";
    for (const auto& kv : res.bond_lengths_angstrom) {
      std::cout << " Bond " << std::left << std::setw(14) << kv.first
                << ": " << std::fixed << std::setprecision(6) << kv.second << " Angstrom\n";
    }
    std::cout << "============================================================================\n";
  }

  return res;
}

VibThermoResult HPCGeometryOptimizer::compute_frequencies(
    const Molecule& mol,
    const GeomOptConfig& cfg,
    double temperature_k,
    double pressure_atm) {

  (void)pressure_atm;
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

  double e0 = evaluate_single_point_cpp(mol, cfg);
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(ndof, ndof);
  const double h = 2.0e-3;
  const double inv_h2 = 1.0 / (h * h);
  const int n_indep = 3 * std::max(0, natoms - 1);

  std::vector<double> ep_1d(n_indep, 0.0);
  std::vector<double> em_1d(n_indep, 0.0);

  // 1. Evaluasi 1D (+h, -h) untuk elemen diagonal H_{kk}
  for (int k = 0; k < n_indep; ++k) {
    Eigen::VectorXd xp = x0, xm = x0;
    xp(k) += h;
    xm(k) -= h;
    ep_1d[k] = evaluate_single_point_cpp(make_displaced_molecule(mol, xp), cfg);
    em_1d[k] = evaluate_single_point_cpp(make_displaced_molecule(mol, xm), cfg);
    hess(k, k) = (ep_1d[k] - 2.0 * e0 + em_1d[k]) * inv_h2;
  }

  // 2. Evaluasi 2D simetris (+h,+h) dan (-h,-h) untuk elemen non-diagonal H_{kl} (k < l)
  for (int k = 0; k < n_indep; ++k) {
    for (int l = k + 1; l < n_indep; ++l) {
      Eigen::VectorXd xpp = x0, xmm = x0;
      xpp(k) += h; xpp(l) += h;
      xmm(k) -= h; xmm(l) -= h;
      const double epp = evaluate_single_point_cpp(make_displaced_molecule(mol, xpp), cfg);
      const double emm = evaluate_single_point_cpp(make_displaced_molecule(mol, xmm), cfg);
      const double h_kl = 0.5 * (epp + emm - ep_1d[k] - em_1d[k] - ep_1d[l] - em_1d[l] + 2.0 * e0) * inv_h2;
      hess(k, l) = h_kl;
      hess(l, k) = h_kl;
    }
  }

  // 3. Rekonstruksi baris & kolom atom terakhir dari invarian translasi eksak (Hukum III Newton)
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
  hess = 0.5 * (hess + hess.transpose());

  VibThermoResult out;
  out.electronic_energy_ha = e0;
  out.hessian_ha_bohr2 = hess;
  out.temperature_k = temperature_k;

  Eigen::MatrixXd V_int = build_internal_vibrational_subspace(x0, masses, true);
  int n_vib = static_cast<int>(V_int.cols());
  int n_tr = ndof - n_vib;

  Eigen::MatrixXd F_mw(ndof, ndof);
  for (int i = 0; i < ndof; ++i) {
    double mi = std::sqrt(masses[i / 3]);
    for (int j = 0; j < ndof; ++j) {
      double mj = std::sqrt(masses[j / 3]);
      F_mw(i, j) = hess(i, j) / (mi * mj);
    }
  }

  Eigen::MatrixXd F_int = V_int.transpose() * F_mw * V_int;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (F_int + F_int.transpose()));
  Eigen::VectorXd evals = es.eigenvalues();
  out.normal_modes = V_int * es.eigenvectors();
  out.frequencies_cm1.resize(n_vib);

  const double force_conv = HARTREE_TO_J / (BOHR_TO_M * BOHR_TO_M * AMU_TO_KG);
  const double to_cm1 = 1.0 / (2.0 * PI * SPEED_OF_LIGHT_CMS);

  double zpe_j = 0.0;
  double u_vib_j = 0.0;
  double s_vib_jk = 0.0;

  for (int k = 0; k < n_vib; ++k) {
    double lam = evals(k);
    if (lam < 0.0) {
      out.frequencies_cm1(k) = -std::sqrt(-lam * force_conv) * to_cm1;
      out.n_imaginary++;
    } else {
      double f_cm = std::sqrt(lam * force_conv) * to_cm1;
      out.frequencies_cm1(k) = f_cm;
      double hv = PLANCK_SI * f_cm * SPEED_OF_LIGHT_CMS;
      zpe_j += 0.5 * hv;
      if (temperature_k > 1.0e-3 && f_cm > 1.0) {
        double x_val = hv / (BOLTZMANN_SI * temperature_k);
        if (x_val < 100.0) {
          u_vib_j += hv / (std::exp(x_val) - 1.0);
          s_vib_jk += BOLTZMANN_SI * (x_val / (std::exp(x_val) - 1.0) - std::log(1.0 - std::exp(-x_val)));
        }
      }
    }
  }

  out.zpe_hartree = zpe_j / HARTREE_TO_J;
  out.zpe_kcal_mol = out.zpe_hartree * HARTREE_TO_KCAL;
  double kt_ha = (BOLTZMANN_SI * temperature_k) / HARTREE_TO_J;
  bool is_linear = (n_tr == 5);
  double u_tr_ha = (is_linear ? 2.5 : 3.0) * kt_ha;

  out.thermal_energy_ha = e0 + out.zpe_hartree + u_tr_ha + (u_vib_j / HARTREE_TO_J);
  out.enthalpy_ha = out.thermal_energy_ha + kt_ha;
  double s_elec_jk = BOLTZMANN_SI * std::log(std::max(1, mol.multiplicity()));
  double s_tr_jk = BOLTZMANN_SI * (is_linear ? 18.5 : 21.0);
  out.gibbs_free_energy_ha = out.enthalpy_ha - temperature_k * (s_tr_jk + s_vib_jk + s_elec_jk) / HARTREE_TO_J;

  if (cfg.verbose) {
    std::cout << "========================================================================\n";
    std::cout << " MSHQC HARMONIC VIBRATIONAL & THERMOCHEMISTRY (" << cfg.method << " / " << cfg.eri_method << ")\n";
    std::cout << "------------------------------------------------------------------------\n";
    for (int k = 0; k < n_vib; ++k) {
      std::cout << " Mode " << std::setw(2) << (k + 1) << ": "
                << std::fixed << std::setprecision(2) << std::setw(10) << out.frequencies_cm1(k) << " cm^-1\n";
    }
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " Zero-Point Energy (ZPE) : " << std::setprecision(8) << out.zpe_hartree
              << " Ha (" << std::setprecision(3) << out.zpe_kcal_mol << " kcal/mol)\n";
    std::cout << " Thermal Enthalpy H(T)   : " << std::setprecision(8) << out.enthalpy_ha << " Ha\n";
    std::cout << " Gibbs Free Energy G(T)  : " << std::setprecision(8) << out.gibbs_free_energy_ha << " Ha\n";
    std::cout << "========================================================================\n";
  }

  return out;
}

std::vector<PESScanPoint> HPCGeometryOptimizer::scan_bond_pes(
    const Molecule& mol,
    int atom_i,
    int atom_j,
    const std::vector<double>& distances_angstrom,
    const GeomOptConfig& cfg) {

  const int natoms = static_cast<int>(mol.n_atoms());
  Eigen::VectorXd x0(3 * natoms);
  for (int i = 0; i < natoms; ++i) {
    x0(3 * i + 0) = mol.atom(i).x;
    x0(3 * i + 1) = mol.atom(i).y;
    x0(3 * i + 2) = mol.atom(i).z;
  }

  Eigen::Vector3d vij = x0.segment<3>(3 * atom_j) - x0.segment<3>(3 * atom_i);
  double r0 = vij.norm();
  Eigen::Vector3d u_ij = (r0 > 1.0e-8) ? (vij / r0) : Eigen::Vector3d(0.0, 0.0, 1.0);

  std::vector<PESScanPoint> pts;
  for (double r_ang : distances_angstrom) {
    double r_bohr = r_ang * ANGSTROM_TO_BOHR;
    Eigen::VectorXd x_scan = x0;
    x_scan.segment<3>(3 * atom_j) = x0.segment<3>(3 * atom_i) + r_bohr * u_ij;
    double e_val = evaluate_single_point_cpp(make_displaced_molecule(mol, x_scan), cfg);
    pts.push_back({r_ang, r_bohr, e_val});
  }
  return pts;
}

OrbitalOptimizedDensityAO build_rhf_omp_ao_densities(
    const Eigen::MatrixXd& C_mo,
    const Eigen::MatrixXd& gamma_mo_total,
    const Eigen::MatrixXd& F_gen_mo
) {
  OrbitalOptimizedDensityAO out;
  Eigen::MatrixXd gamma_sym = 0.5 * (gamma_mo_total + gamma_mo_total.transpose());
  out.P_eff_ao.noalias() = C_mo * gamma_sym * C_mo.transpose();

  Eigen::MatrixXd F_gen_sym = 0.5 * (F_gen_mo + F_gen_mo.transpose());
  out.W_eff_ao.noalias() = C_mo * F_gen_sym * C_mo.transpose();
  return out;
}

OrbitalOptimizedDensityAO build_uhf_omp_ao_densities(
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    const Eigen::MatrixXd& gamma_alpha_mo,
    const Eigen::MatrixXd& gamma_beta_mo,
    const Eigen::MatrixXd& F_gen_alpha_mo,
    const Eigen::MatrixXd& F_gen_beta_mo
) {
  OrbitalOptimizedDensityAO out;
  Eigen::MatrixXd ga_sym = 0.5 * (gamma_alpha_mo + gamma_alpha_mo.transpose());
  Eigen::MatrixXd gb_sym = 0.5 * (gamma_beta_mo + gamma_beta_mo.transpose());

  out.P_eff_ao.noalias() = C_alpha * ga_sym * C_alpha.transpose();
  out.P_eff_ao.noalias() += C_beta * gb_sym * C_beta.transpose();

  Eigen::MatrixXd Fa_sym = 0.5 * (F_gen_alpha_mo + F_gen_alpha_mo.transpose());
  Eigen::MatrixXd Fb_sym = 0.5 * (F_gen_beta_mo + F_gen_beta_mo.transpose());

  out.W_eff_ao.noalias() = C_alpha * Fa_sym * C_alpha.transpose();
  out.W_eff_ao.noalias() += C_beta * Fb_sym * C_beta.transpose();
  return out;
}

} // namespace gradient
} // namespace mshqc
