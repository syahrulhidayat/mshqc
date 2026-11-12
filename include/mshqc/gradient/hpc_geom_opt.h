// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
#ifndef MSHQC_GRADIENT_HPC_GEOM_OPT_H
#define MSHQC_GRADIENT_HPC_GEOM_OPT_H

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>
#include <map>

#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

struct GeomOptConfig {
    std::string method = "omp2";       // "rhf", "uhf", "rohf", "mp2", "rmp2", "ump2", "romp2", "mp3", "rmp3", "ump3", "romp3", "omp2", "omp3"
    std::string eri_method = "df";     // "exact", "cholesky", "df"
    std::string basis_name = "cc-pVDZ";
    std::string aux_basis_name = "";
    std::string reference = "auto";    // "auto", "rhf", "uhf", "rohf"
    std::string opt_method = "soscf";
    int max_iter = 40;
    double energy_thresh = 1.0e-10;
    double orb_grad_thresh = 1.0e-7;
    double g_rms_tol = 1.0e-5;         // Ha / Bohr
    double g_max_tol = 3.0e-5;         // Ha / Bohr
    double dx_rms_tol = 4.0e-4;        // Bohr
    double dx_max_tol = 1.2e-3;        // Bohr
    double trust_radius_init = 0.25;   // Bohr
    double trust_radius_max = 0.50;    // Bohr
    double trust_radius_min = 1.0e-3;  // Bohr
    double fd_step_bohr = 1.0e-3;      // Langkah finite-difference inti (Bohr)
    bool project_trans_rot = true;
    bool verbose = true;
    std::string xyz_traj_file = "mshqc_opt_traj.xyz";
};

struct GeomOptResult {
    bool converged = false;
    int iterations = 0;
    std::string method = "omp2";
    std::string eri_method = "df";
    std::string basis_name = "cc-pVDZ";
    double initial_energy = 0.0;
    double final_energy = 0.0;
    double final_g_rms = 0.0;
    double final_g_max = 0.0;
    double wall_time_sec = 0.0;
    Eigen::MatrixXd coordinates_bohr;     // (N_atoms, 3)
    Eigen::MatrixXd coordinates_angstrom; // (N_atoms, 3)
    Eigen::MatrixXd gradient_ha_bohr;     // (N_atoms, 3)
    std::vector<double> energy_history;
    std::vector<double> g_rms_history;
    std::map<std::string, double> bond_lengths_angstrom;
};

struct VibThermoResult {
    Eigen::VectorXd frequencies_cm1;
    Eigen::MatrixXd normal_modes;
    Eigen::MatrixXd hessian_ha_bohr2;
    double electronic_energy_ha = 0.0;
    double zpe_hartree = 0.0;
    double zpe_kcal_mol = 0.0;
    double thermal_energy_ha = 0.0;
    double enthalpy_ha = 0.0;
    double gibbs_free_energy_ha = 0.0;
    double temperature_k = 298.15;
    int n_imaginary = 0;
};

struct PESScanPoint {
    double r_angstrom = 0.0;
    double r_bohr = 0.0;
    double energy_ha = 0.0;
};

class HPCGeometryOptimizer {
public:
    static double evaluate_single_point_cpp(
        const Molecule& mol,
        const GeomOptConfig& cfg);

    static std::pair<double, Eigen::VectorXd> compute_nuclear_gradient_cpp(
        const Molecule& mol,
        const GeomOptConfig& cfg);

    static Eigen::VectorXd project_out_trans_rot(
        const Eigen::VectorXd& coords_bohr,
        const Eigen::VectorXd& grad,
        const std::vector<double>& masses_amu);

    static GeomOptResult optimize_molecule(
        Molecule& mol,
        const GeomOptConfig& cfg = GeomOptConfig());

    static VibThermoResult compute_frequencies(
        const Molecule& mol,
        const GeomOptConfig& cfg = GeomOptConfig(),
        double temperature_k = 298.15,
        double pressure_atm = 1.0);

    static std::vector<PESScanPoint> scan_bond_pes(
        const Molecule& mol,
        int atom_i,
        int atom_j,
        const std::vector<double>& distances_angstrom,
        const GeomOptConfig& cfg = GeomOptConfig());
};


struct OrbitalOptimizedDensityAO {
    Eigen::MatrixXd P_eff_ao; 
    Eigen::MatrixXd W_eff_ao; 
};


OrbitalOptimizedDensityAO build_rhf_omp_ao_densities(
    const Eigen::MatrixXd& C_mo,
    const Eigen::MatrixXd& gamma_mo_total,
    const Eigen::MatrixXd& F_gen_mo
);

OrbitalOptimizedDensityAO build_uhf_omp_ao_densities(
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    const Eigen::MatrixXd& gamma_alpha_mo,
    const Eigen::MatrixXd& gamma_beta_mo,
    const Eigen::MatrixXd& F_gen_alpha_mo,
    const Eigen::MatrixXd& F_gen_beta_mo
);

} // namespace gradient
} // namespace mshqc

#endif // MSHQC_GRADIENT_HPC_GEOM_OPT_H
