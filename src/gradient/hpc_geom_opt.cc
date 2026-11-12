#include "mshqc/gradient/hpc_geom_opt.h"
#include <fstream>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {
namespace gradient {

namespace {
constexpr double BOHR_TO_ANGSTROM = 0.529177210903;
constexpr double HARTREE_TO_J = 4.3597447222071e-18;
constexpr double BOHR_TO_M = 5.29177210903e-11;
constexpr double AMU_TO_KG = 1.66053906660e-27;
constexpr double SPEED_OF_LIGHT_CMS = 2.99792458e10;
constexpr double PLANCK_SI = 6.62607015e-34;
constexpr double BOLTZMANN_SI = 1.380649e-23;
constexpr double AVOGADRO = 6.02214076e23;
constexpr double PI = 3.14159265358979323846;
} // namespace

Eigen::VectorXd HPCGeometryOptimizer::project_out_trans_rot(
    const Eigen::VectorXd& coords_bohr,
    const Eigen::VectorXd& grad,
    const std::vector<double>& masses_amu) {

    const int natoms = static_cast<int>(masses_amu.size());
    const int ndof = 3 * natoms;
    if (coords_bohr.size() != ndof || grad.size() != ndof) return grad;

    // Hitung pusat massa
    double total_mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < natoms; ++i) {
        double m = (masses_amu[i] > 0.0) ? masses_amu[i] : 1.0;
        total_mass += m;
        com += m * coords_bohr.segment<3>(3 * i);
    }
    com /= total_mass;

    // Bangun 6 vektor basis translasi dan rotasi di ruang Kartesian
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ndof, 6);
    for (int i = 0; i < natoms; ++i) {
        Eigen::Vector3d r = coords_bohr.segment<3>(3 * i) - com;
        // Translasi X, Y, Z
        D(3 * i + 0, 0) = 1.0;
        D(3 * i + 1, 1) = 1.0;
        D(3 * i + 2, 2) = 1.0;
        // Rotasi Rx, Ry, Rz
        D(3 * i + 1, 3) = -r(2);
        D(3 * i + 2, 3) =  r(1);
        D(3 * i + 0, 4) =  r(2);
        D(3 * i + 2, 4) = -r(0);
        D(3 * i + 0, 5) = -r(1);
        D(3 * i + 1, 5) =  r(0);
    }

    // Ortogonalisasi SVD untuk menangani molekul linier maupun non-linier secara eksak
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

GeomOptResult HPCGeometryOptimizer::optimize_tr_bfgs(
    const Eigen::VectorXd& init_coords_bohr,
    const std::vector<double>& masses_amu,
    const std::vector<std::string>& symbols,
    EnergyGradFunc eval_fn,
    const GeomOptConfig& cfg) {

    const int ndof = static_cast<int>(init_coords_bohr.size());
    const int natoms = ndof / 3;

    GeomOptResult res;
    Eigen::VectorXd x = init_coords_bohr;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(ndof, ndof) * 0.5; // Lindh/Schlegel initial scale

    // Inisialisasi pendekatan diagonal Hessian berdasarkan jarak antar-atom
    for (int i = 0; i < natoms; ++i) {
        double diag_stiff = 0.25;
        for (int j = 0; j < natoms; ++j) {
            if (i == j) continue;
            double r_ij = (x.segment<3>(3 * i) - x.segment<3>(3 * j)).norm();
            if (r_ij < 4.0) {
                diag_stiff += 0.35 * std::exp(-0.5 * (r_ij - 2.0));
            }
        }
        H(3 * i + 0, 3 * i + 0) = diag_stiff;
        H(3 * i + 1, 3 * i + 1) = diag_stiff;
        H(3 * i + 2, 3 * i + 2) = diag_stiff;
    }

    double trust_radius = cfg.trust_radius_init;
    Eigen::VectorXd x_prev = x;
    Eigen::VectorXd g_prev = Eigen::VectorXd::Zero(ndof);
    double e_prev = 0.0;

    if (!cfg.xyz_traj_file.empty()) {
        std::ofstream ofs(cfg.xyz_traj_file, std::ios::trunc);
    }

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        auto [energy, raw_grad] = eval_fn(x);
        Eigen::VectorXd g = cfg.project_trans_rot
            ? project_out_trans_rot(x, raw_grad, masses_amu)
            : raw_grad;

        double g_rms = std::sqrt(g.squaredNorm() / static_cast<double>(ndof));
        double g_max = g.cwiseAbs().maxCoeff();

        res.energy_history.push_back(energy);
        res.g_rms_history.push_back(g_rms);

        // Simpan frame ke file XYZ trajectory
        if (!cfg.xyz_traj_file.empty() && static_cast<int>(symbols.size()) == natoms) {
            std::ofstream ofs(cfg.xyz_traj_file, std::ios::app);
            ofs << natoms << "\n";
            ofs << std::fixed << std::setprecision(10)
                << "Step " << iter << " E = " << energy << " Ha | g_rms = " << g_rms << "\n";
            for (int a = 0; a < natoms; ++a) {
                ofs << std::left << std::setw(4) << symbols[a]
                    << std::right << std::setw(16) << x(3 * a + 0) * BOHR_TO_ANGSTROM
                    << std::setw(16) << x(3 * a + 1) * BOHR_TO_ANGSTROM
                    << std::setw(16) << x(3 * a + 2) * BOHR_TO_ANGSTROM << "\n";
            }
        }

        // Update Hessian dengan rumus damped-BFGS (Powell-Murtagh-Sargent)
        if (iter > 0) {
            Eigen::VectorXd s = x - x_prev;
            Eigen::VectorXd y = g - g_prev;
            double sy = s.dot(y);
            Eigen::VectorXd Hs = H * s;
            double sHs = s.dot(Hs);

            // Powell damping untuk menjaga sifat positif-definit pada permukaan anharmonik
            double theta = 1.0;
            if (sy < 0.2 * sHs && sHs > 1.0e-14) {
                theta = (0.8 * sHs) / (sHs - sy);
            }
            Eigen::VectorXd r = theta * y + (1.0 - theta) * Hs;
            double sr = s.dot(r);

            if (sr > 1.0e-12 && sHs > 1.0e-12) {
                H += (r * r.transpose()) / sr - (Hs * Hs.transpose()) / sHs;
            }

            // Adaptasi Trust Radius berdasarkan rasio penurunan energi aktual vs prediksi kuadratik
            double actual_de = energy - e_prev;
            double pred_de = g_prev.dot(s) + 0.5 * s.dot(Hs);
            if (std::abs(pred_de) > 1.0e-12) {
                double ratio = actual_de / pred_de;
                if (ratio > 0.75 && s.norm() >= 0.9 * trust_radius) {
                    trust_radius = std::min(cfg.trust_radius_max, 1.5 * trust_radius);
                } else if (ratio < 0.25) {
                    trust_radius = std::max(cfg.trust_radius_min, 0.5 * trust_radius);
                }
            }
        }

        // Hitung langkah kuasi-Newton dengan Rational Function Optimization (RFO) / Levenberg shift
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        Eigen::VectorXd evals = es.eigenvalues();
        Eigen::MatrixXd evecs = es.eigenvectors();

        Eigen::VectorXd g_mode = evecs.transpose() * g;
        Eigen::VectorXd step_mode(ndof);

        double shift = 0.0;
        double min_eval = evals.minCoeff();
        if (min_eval < 1.0e-4) {
            shift = -min_eval + 1.0e-3;
        }

        for (int k = 0; k < ndof; ++k) {
            double denom = std::max(evals(k) + shift, 1.0e-4);
            step_mode(k) = -g_mode(k) / denom;
        }

        Eigen::VectorXd dx = evecs * step_mode;
        if (cfg.project_trans_rot) {
            dx = project_out_trans_rot(x, dx, masses_amu);
        }

        // Batasi dengan Trust Radius
        double step_norm = dx.norm();
        if (step_norm > trust_radius) {
            dx *= (trust_radius / step_norm);
        }

        double dx_rms = std::sqrt(dx.squaredNorm() / static_cast<double>(ndof));
        double dx_max = dx.cwiseAbs().maxCoeff();

        if (cfg.verbose) {
            std::cout << std::fixed << std::setprecision(10)
                      << "  [OPT] Iter " << std::setw(3) << iter
                      << " | E = " << std::setw(16) << energy
                      << " | g_rms = " << std::scientific << std::setprecision(3) << g_rms
                      << " | g_max = " << g_max
                      << " | dx_rms = " << dx_rms << std::fixed << "\n";
        }

        bool grad_conv = (g_rms < cfg.g_rms_tol) && (g_max < cfg.g_max_tol);
        bool step_conv = (dx_rms < cfg.dx_rms_tol) && (dx_max < cfg.dx_max_tol);
        bool very_tight_grad = (g_rms < 0.2 * cfg.g_rms_tol) && (g_max < 0.2 * cfg.g_max_tol);

        if ((grad_conv && step_conv) || very_tight_grad) {
            res.converged = true;
            res.iterations = iter + 1;
            res.final_energy = energy;
            res.final_g_rms = g_rms;
            res.final_g_max = g_max;
            res.final_coords_bohr = x;
            res.final_gradient = g;
            return res;
        }

        x_prev = x;
        g_prev = g;
        e_prev = energy;
        x += dx;
    }

    res.converged = false;
    res.iterations = cfg.max_iter;
    res.final_energy = res.energy_history.empty() ? 0.0 : res.energy_history.back();
    res.final_g_rms = res.g_rms_history.empty() ? 0.0 : res.g_rms_history.back();
    res.final_coords_bohr = x;
    return res;
}

VibThermoResult HPCGeometryOptimizer::compute_harmonic_frequencies(
    const Eigen::VectorXd& eq_coords_bohr,
    const std::vector<double>& masses_amu,
    const Eigen::MatrixXd& hessian_hartree_bohr2,
    double temperature_k,
    double pressure_atm,
    int spin_multiplicity) {

    const int natoms = static_cast<int>(masses_amu.size());
    const int ndof = 3 * natoms;
    VibThermoResult out;

    // 1. Bangun proyektor P = I - D D^T di ruang terbobot-massa (mass-weighted coordinates)
    double total_mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (int i = 0; i < natoms; ++i) {
        total_mass += masses_amu[i];
        com += masses_amu[i] * eq_coords_bohr.segment<3>(3 * i);
    }
    com /= total_mass;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ndof, 6);
    for (int i = 0; i < natoms; ++i) {
        double sqrt_m = std::sqrt(std::max(masses_amu[i], 1.0e-6));
        Eigen::Vector3d r = eq_coords_bohr.segment<3>(3 * i) - com;
        D(3 * i + 0, 0) = sqrt_m;
        D(3 * i + 1, 1) = sqrt_m;
        D(3 * i + 2, 2) = sqrt_m;

        D(3 * i + 1, 3) = -sqrt_m * r(2);
        D(3 * i + 2, 3) =  sqrt_m * r(1);
        D(3 * i + 0, 4) =  sqrt_m * r(2);
        D(3 * i + 2, 4) = -sqrt_m * r(0);
        D(3 * i + 0, 5) = -sqrt_m * r(1);
        D(3 * i + 1, 5) =  sqrt_m * r(0);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullU);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::VectorXd s = svd.singularValues();
    int n_tr = 0;
    for (int k = 0; k < s.size(); ++k) {
        if (s(k) > 1.0e-6) ++n_tr;
    }

    int n_vib = std::max(0, ndof - n_tr);
    Eigen::MatrixXd V_int = U.rightCols(n_vib); // Sub-ruang vibrasi murni

    // 2. Mass-weight Hessian: F_mw(i, j) = F(i, j) / sqrt(m_i * m_j)
    Eigen::MatrixXd F_mw(ndof, ndof);
    for (int i = 0; i < ndof; ++i) {
        double mi = std::sqrt(std::max(masses_amu[i / 3], 1.0e-6));
        for (int j = 0; j < ndof; ++j) {
            double mj = std::sqrt(std::max(masses_amu[j / 3], 1.0e-6));
            F_mw(i, j) = 0.5 * (hessian_hartree_bohr2(i, j) + hessian_hartree_bohr2(j, i)) / (mi * mj);
        }
    }

    // 3. Proyeksikan ke ruang internal vibrasi & diagonalisasi
    Eigen::MatrixXd F_int = V_int.transpose() * F_mw * V_int;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_int);
    Eigen::VectorXd evals = es.eigenvalues();
    out.normal_modes = V_int * es.eigenvectors();
    out.frequencies_cm1.resize(n_vib);

    // Faktor konversi nilai eigen (Hartree / (Bohr^2 * amu)) ke cm^-1
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
            double freq_cm1 = std::sqrt(lam * force_conv) * to_cm1;
            out.frequencies_cm1(k) = freq_cm1;

            double nu_hz = freq_cm1 * SPEED_OF_LIGHT_CMS;
            double hv = PLANCK_SI * nu_hz;
            zpe_j += 0.5 * hv;

            if (temperature_k > 1.0e-3 && freq_cm1 > 1.0) {
                double x = hv / (BOLTZMANN_SI * temperature_k);
                if (x < 100.0) {
                    double ex = std::exp(x);
                    u_vib_j += hv / (ex - 1.0);
                    s_vib_jk += BOLTZMANN_SI * (x / (ex - 1.0) - std::log(1.0 - std::exp(-x)));
                }
            }
        }
    }

    out.zpe_hartree = zpe_j / HARTREE_TO_J;
    double kt_ha = (BOLTZMANN_SI * temperature_k) / HARTREE_TO_J;
    bool is_linear = (n_tr == 5);
    double u_trans_rot_ha = is_linear ? (2.5 * kt_ha) : (3.0 * kt_ha);

    out.thermal_energy_corr = out.zpe_hartree + u_trans_rot_ha + (u_vib_j / HARTREE_TO_J);
    out.enthalpy_corr = out.thermal_energy_corr + kt_ha;

    // Entropi translasi (Sackur-Tetrode), rotasi, elektronik (spin), dan vibrasi
    double p_pa = pressure_atm * 101325.0;
    double mass_kg = total_mass * AMU_TO_KG;
    double lambda_th = PLANCK_SI / std::sqrt(2.0 * PI * mass_kg * BOLTZMANN_SI * temperature_k);
    double v_mol = (BOLTZMANN_SI * temperature_k) / p_pa;
    double s_trans_jk = BOLTZMANN_SI * (std::log(v_mol / std::pow(lambda_th, 3)) + 2.5);
    double s_elec_jk = BOLTZMANN_SI * std::log(std::max(1, spin_multiplicity));
    double s_rot_jk = (natoms > 1) ? (is_linear ? 1.0 : 1.5) * BOLTZMANN_SI * 8.0 : 0.0;

    double total_ts_ha = temperature_k * (s_trans_jk + s_rot_jk + s_vib_jk + s_elec_jk) / HARTREE_TO_J;
    out.gibbs_free_energy_corr = out.enthalpy_corr - total_ts_ha;

    return out;
}

} // namespace gradient
} // namespace mshqc
