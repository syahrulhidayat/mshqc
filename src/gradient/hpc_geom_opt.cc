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

// Peredam std::cout RAII agar output internal DF/SCF tidak membanjiri layar selama evaluasi gradien
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
        {1, 1.007825032}, {2, 4.002603254}, {3, 7.016003437}, {4, 9.01218307},
        {5, 11.00930536}, {6, 12.00000000}, {7, 14.003074004}, {8, 15.994914619},
        {9, 18.998403163}, {10, 19.99244018}, {11, 22.98976928}, {12, 23.98504170},
        {13, 26.98153853}, {14, 27.976926535}, {15, 30.973761998}, {16, 31.972071174},
        {17, 34.96885268}, {18, 39.96238312}, {35, 78.918338}, {53, 126.904472}
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

} // namespace

double HPCGeometryOptimizer::evaluate_single_point_cpp(
    const Molecule& mol,
    const GeomOptConfig& cfg) {

    ScopedCoutSilencer silencer(true);

    std::string method = to_lower_copy(cfg.method);
    std::string eri_mode = to_lower_copy(cfg.eri_method);
    if (eri_mode == "ri") eri_mode = "df";

    // Parse prefix seperti "df-omp2" atau "cholesky-omp3"
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
    scf_conf.energy_threshold = cfg.energy_thresh;
    scf_conf.density_threshold = std::min(1.0e-9, cfg.energy_thresh * 10.0);
    scf_conf.cholesky_threshold = cfg.energy_thresh;
    scf_conf.df_threshold = cfg.energy_thresh;
    scf_conf.print_level = 0;
    scf_conf.max_iterations = 150;

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
    mp2_conf.energy_threshold = cfg.energy_thresh;
    mp2_conf.df_threshold = cfg.energy_thresh;
    mp2_conf.cholesky_threshold = cfg.energy_thresh;
    mp2_conf.gradient_threshold = cfg.orb_grad_thresh;
    mp2_conf.opt_method = cfg.opt_method;
    mp2_conf.print_level = 0;
    mp2_conf.max_iterations = 80;

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
        Eigen::Vector3d r = coords_bohr.segment<3>(3 * i) - com;
        D(3 * i + 0, 0) = 1.0;
        D(3 * i + 1, 1) = 1.0;
        D(3 * i + 2, 2) = 1.0;

        D(3 * i + 1, 3) = -r(2);
        D(3 * i + 2, 3) =  r(1);
        D(3 * i + 0, 4) =  r(2);
        D(3 * i + 2, 4) = -r(0);
        D(3 * i + 0, 5) = -r(1);
        D(3 * i + 1, 5) =  r(0);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeThinU);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::VectorXd s = svd.singularValues();

    Eigen::VectorXd g_proj = grad;
    for (int k = 0; k < s.size(); ++k) {
        if (s(k) > 1.0e-7) {
            Eigen::VectorXd uk = U.col(k);
            g_proj -= uk.dot(g_proj) * uk;
        }
    }
    return g_proj;
}

std::pair<double, Eigen::VectorXd> HPCGeometryOptimizer::compute_nuclear_gradient_cpp(
    const Molecule& mol,
    const GeomOptConfig& cfg) {

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
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(ndof);
    if (natoms <= 1) {
        return {e0, grad};
    }

    const double h = cfg.fd_step_bohr;

    // Eksploitasi Invarian Translasi (Hukum III Newton):
    // Evaluasi hanya (N - 1) atom pertama, lalu F_last = -sum_{a=0}^{N-2} F_a.
    // Untuk molekul diatomik (N=2), ini menghemat waktu komputasi tepat 50%!
    for (int a = 0; a < natoms - 1; ++a) {
        for (int xyz = 0; xyz < 3; ++xyz) {
            // Deteksi bidang/sumbu simetri Kartesian: jika seluruh atom berada di bidang xyz == 0,
            // gaya tegak lurus bidang cermin tersebut secara analitik identik 0!
            bool all_on_mirror_plane = true;
            for (int b = 0; b < natoms; ++b) {
                if (std::abs(x0(3 * b + xyz)) > 1.0e-11) {
                    all_on_mirror_plane = false;
                    break;
                }
            }
            if (all_on_mirror_plane) {
                grad(3 * a + xyz) = 0.0;
                continue;
            }

            Eigen::VectorXd xp = x0;
            Eigen::VectorXd xm = x0;
            xp(3 * a + xyz) += h;
            xm(3 * a + xyz) -= h;

            double ep = evaluate_single_point_cpp(make_displaced_molecule(mol, xp), cfg);
            double em = evaluate_single_point_cpp(make_displaced_molecule(mol, xm), cfg);
            grad(3 * a + xyz) = (ep - em) / (2.0 * h);
        }
    }

    // Suku atom terakhir dari invarian translasi
    for (int xyz = 0; xyz < 3; ++xyz) {
        double sum_other = 0.0;
        for (int a = 0; a < natoms - 1; ++a) {
            sum_other += grad(3 * a + xyz);
        }
        grad(3 * (natoms - 1) + xyz) = -sum_other;
    }

    if (cfg.project_trans_rot) {
        grad = project_out_trans_rot(x0, grad, masses);
    }

    return {e0, grad};
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

    // Inisialisasi Hessian Model Lindh-Schlegel di ruang ikatan
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(ndof, ndof) * 0.35;
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
    double tr = cfg.trust_radius_init;

    if (cfg.verbose) {
        std::cout << "============================================================================\n";
        std::cout << "  MSHQC C++20 GEOMETRY OPTIMIZER | Method: " << cfg.method
                  << " (" << cfg.eri_method << ") | Basis: " << cfg.basis_name << "\n";
        std::cout << "----------------------------------------------------------------------------\n";
        std::cout << "  Iter |  Total Energy (Ha) |      dE (Ha) |   RMS Grad |   MAX Grad |   Step RMS\n";
        std::cout << "----------------------------------------------------------------------------\n";
    }

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        Molecule cur_mol = make_displaced_molecule(mol, x);
        auto [energy, grad_cur] = compute_nuclear_gradient_cpp(cur_mol, cfg);
        g = grad_cur;

        // Backtracking line-search jika energi melonjak naik
        if (iter > 0 && energy > e_prev + 1.0e-4) {
            tr = std::max(cfg.trust_radius_min, 0.4 * (x - x_prev).norm());
            x = 0.5 * (x + x_prev);
            cur_mol = make_displaced_molecule(mol, x);
            std::tie(energy, g) = compute_nuclear_gradient_cpp(cur_mol, cfg);
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

        // Update Damped-BFGS Hessian
        if (iter > 0) {
            Eigen::VectorXd s = x - x_prev;
            Eigen::VectorXd y = g - g_prev;
            double sy = s.dot(y);
            Eigen::VectorXd Hs = H * s;
            double sHs = s.dot(Hs);

            double theta = 1.0;
            if (sy < 0.2 * sHs && sHs > 1.0e-14) {
                theta = (0.8 * sHs) / (sHs - sy);
            }
            Eigen::VectorXd r_vec = theta * y + (1.0 - theta) * Hs;
            double sr = s.dot(r_vec);

            if (sr > 1.0e-12 && sHs > 1.0e-12) {
                H += (r_vec * r_vec.transpose()) / sr - (Hs * Hs.transpose()) / sHs;
            }

            double pred_de = g_prev.dot(s) + 0.5 * sHs;
            if (std::abs(pred_de) > 1.0e-12) {
                double ratio = de / pred_de;
                if (ratio > 0.75 && s.norm() >= 0.85 * tr) {
                    tr = std::min(cfg.trust_radius_max, 1.4 * tr);
                } else if (ratio < 0.25) {
                    tr = std::max(cfg.trust_radius_min, 0.5 * tr);
                }
            }
        }

        // Langkah RFO / Quasi-Newton di sub-ruang vibrasi
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (H + H.transpose()));
        Eigen::VectorXd evals = es.eigenvalues();
        Eigen::MatrixXd evecs = es.eigenvectors();

        Eigen::VectorXd g_mode = evecs.transpose() * g;
        Eigen::VectorXd dx_mode(ndof);
        double shift = (evals.minCoeff() < 1.0e-3) ? (-evals.minCoeff() + 1.0e-3) : 0.0;
        for (int k = 0; k < ndof; ++k) {
            dx_mode(k) = -g_mode(k) / std::max(evals(k) + shift, 1.0e-3);
        }

        Eigen::VectorXd dx = evecs * dx_mode;
        if (cfg.project_trans_rot) {
            dx = project_out_trans_rot(x, dx, masses);
        }

        double step_norm = dx.norm();
        if (step_norm > tr) {
            dx *= (tr / step_norm);
        }

        double dx_rms = std::sqrt(dx.squaredNorm() / static_cast<double>(ndof));
        double dx_max = dx.cwiseAbs().maxCoeff();

        if (cfg.verbose) {
            std::cout << "  " << std::setw(4) << iter
                      << " | " << std::fixed << std::setprecision(10) << std::setw(18) << energy
                      << " | " << std::scientific << std::setprecision(3) << std::setw(12) << de
                      << " | " << std::setw(10) << g_rms
                      << " | " << std::setw(10) << g_max
                      << " | " << std::setw(10) << dx_rms << std::fixed << "\n";
        }

        bool grad_ok = (g_rms < cfg.g_rms_tol) && (g_max < cfg.g_max_tol);
        bool step_ok = (dx_rms < cfg.dx_rms_tol) && (dx_max < cfg.dx_max_tol);
        bool tight_grad = (g_rms < 0.25 * cfg.g_rms_tol) && (g_max < 0.25 * cfg.g_max_tol);

        if ((grad_ok && step_ok) || tight_grad || natoms <= 1) {
            res.converged = true;
            break;
        }

        x_prev = x;
        g_prev = g;
        e_prev = energy;
        x += dx;
    }

    // Mutasi in-place koordinat molekul ke geometri teroptimasi!
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
        std::cout << "  Status             : " << (res.converged ? "CONVERGED" : "MAX_ITER REACHED")
                  << " in " << res.iterations << " steps (" << std::fixed << std::setprecision(2)
                  << res.wall_time_sec << " s)\n";
        std::cout << "  Initial Energy     : " << std::setprecision(10) << res.initial_energy << " Ha\n";
        std::cout << "  Final Opt Energy   : " << std::setprecision(10) << res.final_energy << " Ha\n";
        std::cout << "  Relaxation Lowering: " << std::setprecision(4)
                  << (res.final_energy - res.initial_energy) * HARTREE_TO_KCAL << " kcal/mol\n";
        for (const auto& kv : res.bond_lengths_angstrom) {
            std::cout << "  Bond " << std::left << std::setw(14) << kv.first
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

    GeomOptConfig grad_cfg = cfg;
    grad_cfg.project_trans_rot = false;
    grad_cfg.verbose = false;

    for (int k = 0; k < ndof; ++k) {
        Eigen::VectorXd xp = x0;
        Eigen::VectorXd xm = x0;
        xp(k) += h;
        xm(k) -= h;
        auto [ep, gp] = compute_nuclear_gradient_cpp(make_displaced_molecule(mol, xp), grad_cfg);
        auto [em, gm] = compute_nuclear_gradient_cpp(make_displaced_molecule(mol, xm), grad_cfg);
        hess.col(k) = (gp - gm) / (2.0 * h);
    }
    hess = 0.5 * (hess + hess.transpose());

    VibThermoResult out;
    out.electronic_energy_ha = e0;
    out.hessian_ha_bohr2 = hess;
    out.temperature_k = temperature_k;

    double total_mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < natoms; ++i) {
        total_mass += masses[i];
        com += masses[i] * x0.segment<3>(3 * i);
    }
    com /= total_mass;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ndof, 6);
    for (int i = 0; i < natoms; ++i) {
        double sm = std::sqrt(masses[i]);
        Eigen::Vector3d r = x0.segment<3>(3 * i) - com;
        D(3 * i + 0, 0) = sm;
        D(3 * i + 1, 1) = sm;
        D(3 * i + 2, 2) = sm;
        D(3 * i + 1, 3) = -sm * r(2);
        D(3 * i + 2, 3) =  sm * r(1);
        D(3 * i + 0, 4) =  sm * r(2);
        D(3 * i + 2, 4) = -sm * r(0);
        D(3 * i + 0, 5) = -sm * r(1);
        D(3 * i + 1, 5) =  sm * r(0);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullU);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::VectorXd s = svd.singularValues();
    int n_tr = 0;
    for (int k = 0; k < s.size(); ++k) {
        if (s(k) > 1.0e-6) ++n_tr;
    }

    int n_vib = std::max(0, ndof - n_tr);
    Eigen::MatrixXd V_int = U.rightCols(n_vib);

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
        std::cout << "  MSHQC HARMONIC VIBRATIONAL & THERMOCHEMISTRY (" << cfg.method << " / " << cfg.eri_method << ")\n";
        std::cout << "------------------------------------------------------------------------\n";
        for (int k = 0; k < n_vib; ++k) {
            std::cout << "    Mode " << std::setw(2) << (k + 1) << ": "
                      << std::fixed << std::setprecision(2) << std::setw(10) << out.frequencies_cm1(k) << " cm^-1\n";
        }
        std::cout << "------------------------------------------------------------------------\n";
        std::cout << "  Zero-Point Energy (ZPE) : " << std::setprecision(8) << out.zpe_hartree
                  << " Ha (" << std::setprecision(3) << out.zpe_kcal_mol << " kcal/mol)\n";
        std::cout << "  Thermal Enthalpy H(T)   : " << std::setprecision(8) << out.enthalpy_ha << " Ha\n";
        std::cout << "  Gibbs Free Energy G(T)  : " << std::setprecision(8) << out.gibbs_free_energy_ha << " Ha\n";
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
