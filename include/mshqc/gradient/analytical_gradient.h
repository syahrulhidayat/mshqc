#ifndef MSHQC_ANALYTICAL_GRADIENT_H
#define MSHQC_ANALYTICAL_GRADIENT_H

#include "mshqc/gradient/gradient.h"
#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/scf/scf.h"
#include <Eigen/Dense>
#include <memory>
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
);

Eigen::MatrixXd compute_core_hamiltonian_derivative(
    const BasisSet& basis,
    const Molecule& mol,
    int atom_idx,
    int coord
);

double compute_eri_derivative_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& density_matrix,
    int atom_idx,
    int coord
);

Eigen::MatrixXd compute_z_vector_rhf(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    int n_occ,
    const Eigen::MatrixXd& F_deriv,
    const Eigen::MatrixXd& S
);

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
);

class RHFAnalyticalGradient {
public:

    RHFAnalyticalGradient(
        const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SCFResult& scf_result
    );

    GradientResult compute();

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_result_;

    Eigen::MatrixXd P_;
    Eigen::MatrixXd W_;

    Eigen::Vector3d compute_atom_gradient(int atom_idx);

    Eigen::Vector3d compute_nuclear_gradient(int atom_idx);

    Eigen::Vector3d compute_electronic_gradient(int atom_idx);
};

class UHFAnalyticalGradient {
public:
    UHFAnalyticalGradient(
        const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SCFResult& scf_result
    );

    GradientResult compute();

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_result_;

    Eigen::MatrixXd P_alpha_;
    Eigen::MatrixXd P_beta_;
    Eigen::MatrixXd W_alpha_;
    Eigen::MatrixXd W_beta_;

    Eigen::Vector3d compute_atom_gradient(int atom_idx);
    Eigen::Vector3d compute_nuclear_gradient(int atom_idx);
    Eigen::Vector3d compute_electronic_gradient(int atom_idx);
};

GradientResult compute_rhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SCFConfig& config = SCFConfig()
);

GradientResult compute_uhf_gradient_analytical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int n_alpha,
    int n_beta,
    const SCFConfig& config = SCFConfig()
);

double validate_analytical_gradient(
    const Molecule& mol,
    const BasisSet& basis,
    const std::string& method,
    double numerical_step = 1e-5
);

}
}

#endif
