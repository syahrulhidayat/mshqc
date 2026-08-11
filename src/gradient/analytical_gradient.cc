// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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
// ==============================================================================

















#include "mshqc/gradient/analytical_gradient.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

Eigen::MatrixXd compute_overlap_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
) {

    size_t nbasis = basis.n_basis_functions();
    Eigen::MatrixXd dS = Eigen::MatrixXd::Zero(nbasis, nbasis);

    std::cerr << "WARNING: compute_overlap_derivative not fully implemented\n";
    std::cerr << "         Using placeholder (returns zero matrix)\n";

    return dS;
}

Eigen::MatrixXd compute_core_hamiltonian_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
) {

    size_t nbasis = basis.n_basis_functions();
    Eigen::MatrixXd dH = Eigen::MatrixXd::Zero(nbasis, nbasis);

    std::cerr << "WARNING: compute_core_hamiltonian_derivative not fully implemented\n";
    std::cerr << "         Using placeholder (returns zero matrix)\n";

    return dH;
}

double compute_eri_derivative_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& density_matrix,
    int atom_idx,
    int coord
) {

    double contrib = 0.0;

    std::cerr << "WARNING: compute_eri_derivative_contribution not fully implemented\n";
    std::cerr << "         Using placeholder (returns 0.0)\n";

    return contrib;
}

Eigen::MatrixXd compute_z_vector_rhf(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    int n_occ,
    const Eigen::MatrixXd& F_deriv,
    const Eigen::MatrixXd& S
) {

    int nbasis = C.rows();
    int nvirt = nbasis - n_occ;

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nbasis, nbasis);

    if (nvirt <= 0) {

        return W;
    }

    Eigen::MatrixXd P = 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
    W = -P * F_deriv * P;

    std::cerr << "WARNING: compute_z_vector_rhf using diagonal approximation\n";
    std::cerr << "         Full CPHF solver not yet implemented\n";

    return W;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> compute_z_vector_uhf(
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    const Eigen::VectorXd& eps_alpha,
    const Eigen::VectorXd& eps_beta,
    int n_alpha,
    int n_beta,
    const Eigen::MatrixXd& F_alpha_deriv,
    const Eigen::MatrixXd& F_beta_deriv,
    const Eigen::MatrixXd& S
) {

    int nbasis = C_alpha.rows();

    Eigen::MatrixXd P_alpha = C_alpha.leftCols(n_alpha) * C_alpha.leftCols(n_alpha).transpose();
    Eigen::MatrixXd P_beta = C_beta.leftCols(n_beta) * C_beta.leftCols(n_beta).transpose();

    Eigen::MatrixXd W_alpha = -P_alpha * F_alpha_deriv * P_alpha;
    Eigen::MatrixXd W_beta = -P_beta * F_beta_deriv * P_beta;

    std::cerr << "WARNING: compute_z_vector_uhf using diagonal approximation\n";

    return {W_alpha, W_beta};
}

RHFAnalyticalGradient::RHFAnalyticalGradient(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFResult& scf_result
) : mol_(mol), basis_(basis), integrals_(integrals), scf_result_(scf_result) {

    int n_occ = scf_result.C_alpha.cols() / 2;
    P_ = 2.0 * scf_result.C_alpha.leftCols(n_occ) *
         scf_result.C_alpha.leftCols(n_occ).transpose();

    W_ = Eigen::MatrixXd::Zero(P_.rows(), P_.cols());
}

GradientResult RHFAnalyticalGradient::compute() {
    int natoms = mol_.n_atoms();
    Eigen::VectorXd gradient(3 * natoms);

    std::cout << "\n========================================\n";
    std::cout << "  Analytical Gradient Calculation (RHF)\n";
    std::cout << "========================================\n";
    std::cout << "Method: RHF analytical\n";
    std::cout << "Atoms:  " << natoms << "\n";
    std::cout << "========================================\n\n";

    for (int atom = 0; atom < natoms; ++atom) {
        Eigen::Vector3d grad_atom = compute_atom_gradient(atom);

        gradient(3*atom + 0) = grad_atom(0);
        gradient(3*atom + 1) = grad_atom(1);
        gradient(3*atom + 2) = grad_atom(2);

        std::cout << "Atom " << (atom+1) << "  "
                  << std::scientific << std::setprecision(6)
                  << grad_atom(0) << "  "
                  << grad_atom(1) << "  "
                  << grad_atom(2) << "\n";
    }

    GradientResult result;
    result.gradient = gradient;
    result.energy = scf_result_.energy_total;
    result.method = "RHF (analytical)";
    result.is_analytical = true;
    result.populate_gradient_by_atom(natoms);

    double rms = std::sqrt(gradient.squaredNorm() / (3 * natoms));
    double max_component = gradient.cwiseAbs().maxCoeff();
    result.rms_gradient = rms;
    result.max_gradient = max_component;

    std::cout << "\n";
    std::cout << "RMS gradient: " << std::scientific << std::setprecision(4)
              << rms << " Ha/bohr\n";
    std::cout << "Max gradient: " << max_component << " Ha/bohr\n";
    std::cout << "========================================\n\n";

    return result;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_atom_gradient(int atom_idx) {
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();

    Eigen::Vector3d grad_nuc = compute_nuclear_gradient(atom_idx);

    Eigen::Vector3d grad_elec = compute_electronic_gradient(atom_idx);

    grad = grad_nuc + grad_elec;

    return grad;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_nuclear_gradient(int atom_idx) {

    Eigen::Vector3d grad_nuc = Eigen::Vector3d::Zero();

    const auto& atom_A = mol_.atom(atom_idx);
    double Z_A = atom_A.atomic_number;
    Eigen::Vector3d R_A(atom_A.x, atom_A.y, atom_A.z);

    int natoms = mol_.n_atoms();
    for (int B = 0; B < natoms; ++B) {
        if (B == atom_idx) continue;

        const auto& atom_B = mol_.atom(B);
        double Z_B = atom_B.atomic_number;
        Eigen::Vector3d R_B(atom_B.x, atom_B.y, atom_B.z);

        Eigen::Vector3d R_AB = R_A - R_B;
        double dist = R_AB.norm();
        double dist3 = dist * dist * dist;

        grad_nuc += Z_A * Z_B * R_AB / dist3;
    }

    return grad_nuc;
}

Eigen::Vector3d RHFAnalyticalGradient::compute_electronic_gradient(int atom_idx) {

    Eigen::Vector3d grad_elec = Eigen::Vector3d::Zero();

    for (int coord = 0; coord < 3; ++coord) {

        Eigen::MatrixXd dS = compute_overlap_derivative(basis_, mol_, atom_idx, coord);
        double overlap_contrib = (W_.array() * dS.array()).sum();

        Eigen::MatrixXd dH = compute_core_hamiltonian_derivative(basis_, mol_, atom_idx, coord);
        double core_contrib = (P_.array() * dH.array()).sum();

        double eri_contrib = compute_eri_derivative_contribution(basis_, mol_, P_, atom_idx, coord);

        grad_elec(coord) = overlap_contrib + core_contrib + eri_contrib;
    }

    return grad_elec;
}

UHFAnalyticalGradient::UHFAnalyticalGradient(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFResult& scf_result
) : mol_(mol), basis_(basis), integrals_(integrals), scf_result_(scf_result) {

    std::cerr << "WARNING: UHFAnalyticalGradient not fully implemented\n";
}

GradientResult UHFAnalyticalGradient::compute() {
    GradientResult result;
    result.method = "UHF (analytical) - NOT IMPLEMENTED";
    result.is_analytical = true;

    std::cerr << "ERROR: UHF analytical gradient not yet implemented\n";

    return result;
}

Eigen::Vector3d UHFAnalyticalGradient::compute_atom_gradient(int atom_idx) {
    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d UHFAnalyticalGradient::compute_nuclear_gradient(int atom_idx) {
    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d UHFAnalyticalGradient::compute_electronic_gradient(int atom_idx) {
    return Eigen::Vector3d::Zero();
}

GradientResult compute_rhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFConfig& config
) {

    RHF rhf(mol, basis, integrals, nullptr, nullptr, config);
    auto scf_result = rhf.compute();

    if (!scf_result.converged) {
        std::cerr << "WARNING: SCF did not converge\n";
    }

    RHFAnalyticalGradient grad_calc(mol, basis, integrals, scf_result);
    return grad_calc.compute();
}

GradientResult compute_uhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int n_alpha,
    int n_beta,
    const SCFConfig& config
) {

    UHF uhf(mol, basis, integrals, nullptr, nullptr, n_alpha, n_beta, config);
    auto scf_result = uhf.compute();

    if (!scf_result.converged) {
        std::cerr << "WARNING: SCF did not converge\n";
    }

    UHFAnalyticalGradient grad_calc(mol, basis, integrals, scf_result);
    return grad_calc.compute();
}

double validate_analytical_gradient(
    const Molecule& mol,
    const BasisSet& basis,
    const std::string& method,
    double numerical_step
) {
    std::cout << "\n========================================\n";
    std::cout << "  Gradient Validation\n";
    std::cout << "========================================\n";
    std::cout << "Method: " << method << "\n";
    std::cout << "Comparing analytical vs numerical\n";
    std::cout << "Numerical step: " << numerical_step << " au\n";
    std::cout << "========================================\n\n";

    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    SCFConfig config;
    config.print_level = 0;

    GradientResult grad_analytical, grad_numerical;

    if (method == "RHF") {
        grad_analytical = compute_rhf_gradient_analytical(mol, basis, integrals, config);
        grad_numerical = compute_rhf_gradient_numerical(mol, basis, integrals, 0, config, numerical_step);
    } else {
        std::cerr << "ERROR: Only RHF validation currently supported\n";
        return -1.0;
    }

    Eigen::VectorXd diff = grad_analytical.gradient - grad_numerical.gradient;
    double max_diff = diff.cwiseAbs().maxCoeff();
    double rms_diff = std::sqrt(diff.squaredNorm() / diff.size());

    std::cout << "\nComparison:\n";
    std::cout << "  Max difference: " << std::scientific << std::setprecision(4)
              << max_diff << " Ha/bohr\n";
    std::cout << "  RMS difference: " << rms_diff << " Ha/bohr\n";
    std::cout << "\n";

    if (max_diff < 1e-5) {
        std::cout << "✓ Analytical gradient VALIDATED (excellent agreement)\n";
    } else if (max_diff < 1e-4) {
        std::cout << "⚠ Analytical gradient acceptable (good agreement)\n";
    } else {
        std::cout << "✗ Analytical gradient FAILED (poor agreement)\n";
        std::cout << "  Note: Integral derivatives not fully implemented\n";
    }

    std::cout << "========================================\n\n";

    return max_diff;
}

}
}
