#pragma once

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace mshqc {
namespace gradient {

struct GeomOptConfig {
    int max_iter = 50;
    double g_rms_tol = 1.0e-5;     // Ha / Bohr
    double g_max_tol = 3.0e-5;     // Ha / Bohr
    double dx_rms_tol = 4.0e-4;    // Bohr
    double dx_max_tol = 1.2e-3;    // Bohr
    double trust_radius_init = 0.30; // Bohr
    double trust_radius_max = 0.50;  // Bohr
    double trust_radius_min = 1.0e-3;
    double fd_step_bohr = 5.0e-4;  // Langkah diferensiasi numerik presisi tinggi
    bool project_trans_rot = true; // Proyeksikan 6 mode translasi & rotasi
    bool verbose = true;
    std::string xyz_traj_file = "mshqc_opt_traj.xyz";
};

struct GeomOptResult {
    bool converged = false;
    int iterations = 0;
    double final_energy = 0.0;
    double final_g_rms = 0.0;
    double final_g_max = 0.0;
    Eigen::VectorXd final_coords_bohr; // 3N vector
    Eigen::VectorXd final_gradient;    // 3N vector (Ha/Bohr)
    std::vector<double> energy_history;
    std::vector<double> g_rms_history;
};

struct VibThermoResult {
    Eigen::VectorXd frequencies_cm1; // Frekuensi harmonik dalam cm^-1
    Eigen::MatrixXd normal_modes;    // Vektor eigen mode normal
    double zpe_hartree = 0.0;        // Zero-Point Vibrational Energy (Ha)
    double thermal_energy_corr = 0.0;
    double enthalpy_corr = 0.0;
    double gibbs_free_energy_corr = 0.0;
    int n_imaginary = 0;
};

class HPCGeometryOptimizer {
public:
    using EnergyGradFunc = std::function<std::pair<double, Eigen::VectorXd>(const Eigen::VectorXd&)>;

    static Eigen::VectorXd project_out_trans_rot(
        const Eigen::VectorXd& coords_bohr,
        const Eigen::VectorXd& grad,
        const std::vector<double>& masses_amu);

    static GeomOptResult optimize_tr_bfgs(
        const Eigen::VectorXd& init_coords_bohr,
        const std::vector<double>& masses_amu,
        const std::vector<std::string>& symbols,
        EnergyGradFunc eval_fn,
        const GeomOptConfig& cfg = GeomOptConfig());

    static VibThermoResult compute_harmonic_frequencies(
        const Eigen::VectorXd& eq_coords_bohr,
        const std::vector<double>& masses_amu,
        const Eigen::MatrixXd& hessian_hartree_bohr2,
        double temperature_k = 298.15,
        double pressure_atm = 1.0,
        int spin_multiplicity = 1);
};

} // namespace gradient
} // namespace mshqc
