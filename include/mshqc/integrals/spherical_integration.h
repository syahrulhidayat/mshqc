#ifndef MSHQC_SPHERICAL_INTEGRATION_H
#define MSHQC_SPHERICAL_INTEGRATION_H

#include "spherical_transformer.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/basis.h"
#include "mshqc/core/molecule.h"
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#ifdef I
#undef I
#endif

namespace mshqc {

class SphericalIntegralEngine {
public:
    SphericalIntegralEngine(
        const BasisSet& basis,
        const Molecule& mol,

        bool force_spherical = true

    );

    Eigen::MatrixXd compute_overlap_spherical();
    Eigen::MatrixXd compute_kinetic_spherical();
    Eigen::MatrixXd compute_nuclear_spherical();

    Eigen::Tensor<double, 4> compute_eri_spherical();

    Eigen::MatrixXd transform_to_spherical(const Eigen::MatrixXd& cart_matrix);
    Eigen::Tensor<double, 4> transform_eri_to_spherical(

        const Eigen::Tensor<double, 4>& cart_eri

    );

    int get_nbf_cartesian() const { return nbf_cart_; }
    int get_nbf_spherical() const { return nbf_sph_; }
    bool is_spherical() const { return is_spherical_; }
    bool needs_transformation() const { return needs_transform_; }

    const SphericalTransformer& get_transformer() const { return transformer_; }
    const BasisTransformationHelper& get_helper() const { return *helper_; }

    void debug_print_offsets() const;

private:
    const BasisSet& basis_;
    const Molecule& mol_;

    SphericalTransformer transformer_;
    std::unique_ptr<BasisTransformationHelper> helper_;

    std::vector<int> angular_momenta_;
    std::vector<int> cart_offsets_;
    std::vector<int> sph_offsets_;

    int nbf_cart_;
    int nbf_sph_;
    bool is_spherical_;
    bool needs_transform_;
    bool force_spherical_;

    void initialize();
    void extract_angular_momenta();
};

class UHFSphericalHelper {
public:
    UHFSphericalHelper(
        const BasisSet& basis,
        const Molecule& mol,
        bool use_spherical = true
    );

    struct UHFIntegrals {
        Eigen::MatrixXd S;

        Eigen::MatrixXd T;

        Eigen::MatrixXd V;

        Eigen::Tensor<double, 4> ERI;

        int nbf;
        bool is_spherical;
    };

    UHFIntegrals prepare_integrals();

    Eigen::MatrixXd transform_density_to_cartesian(const Eigen::MatrixXd& D_sph);
    Eigen::MatrixXd transform_mo_to_cartesian(const Eigen::MatrixXd& C_sph);

private:
    SphericalIntegralEngine engine_;
    const BasisSet& basis_;
    const Molecule& mol_;
    bool use_spherical_;
};

namespace spherical_utils {

bool requires_spherical_transformation(const BasisSet& basis);

void print_transformation_info(
    const BasisTransformationHelper& helper,
    std::ostream& os = std::cout
);

bool validate_transformation(
    const Eigen::MatrixXd& cart_matrix,
    const Eigen::MatrixXd& sph_matrix,
    const SphericalTransformer& transformer,
    const std::vector<int>& angular_momenta,
    double threshold = 1e-10
);

size_t estimate_memory_requirement(
    int nbf_cart,
    int nbf_sph,
    bool store_eri = true
);

}

class QuickSphericalTransform {
public:

    static Eigen::MatrixXd transform_matrix(
        const Eigen::MatrixXd& cart_matrix,
        const BasisSet& basis,
        const Molecule& mol

    );

    static Eigen::Tensor<double, 4> transform_eri(

        const Eigen::Tensor<double, 4>& cart_eri,

        const BasisSet& basis,
        const Molecule& mol

    );

    static Eigen::MatrixXd transform_mo_coefficients(
        const Eigen::MatrixXd& cart_coeff,
        const BasisSet& basis,
        const Molecule& mol

    );

private:
    QuickSphericalTransform() = delete;
};

}

#endif
