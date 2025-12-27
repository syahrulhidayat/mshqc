// spherical_integration.h
// Author: Muhamad Syahrul Hidayat
// Integrasi transformasi spherical dengan sistem SCF/UHF

#ifndef MSHQC_SPHERICAL_INTEGRATION_H
#define MSHQC_SPHERICAL_INTEGRATION_H

#include "spherical_transformer.h"
#include "mshqc/integrals.h"
#include "mshqc/basis.h"
#include "mshqc/molecule.h" 
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor> 
#include <iostream> 

namespace mshqc {

/**
 * @brief Wrapper untuk integrasi spherical transformation dengan sistem
 */
class SphericalIntegralEngine {
public:
    SphericalIntegralEngine(
        const BasisSet& basis,
        const Molecule& mol,  // TAMBAHKAN parameter ini
        bool force_spherical = true
        
    );
    
    
    // Hitung integral dalam basis spherical
    Eigen::MatrixXd compute_overlap_spherical();
    Eigen::MatrixXd compute_kinetic_spherical();
    Eigen::MatrixXd compute_nuclear_spherical();  // HAPUS parameter (const Molecule& mol)
    Eigen::Tensor<double, 4> compute_eri_spherical();  // UBAH dari std::vector<double>
    
    // Transform dari Cartesian yang sudah ada
    Eigen::MatrixXd transform_to_spherical(const Eigen::MatrixXd& cart_matrix);
    Eigen::Tensor<double, 4> transform_eri_to_spherical(  // UBAH dari std::vector<double>
        const Eigen::Tensor<double, 4>& cart_eri  // UBAH dari const std::vector<double>&
    );
    
    // Info basis
    int get_nbf_cartesian() const { return nbf_cart_; }
    int get_nbf_spherical() const { return nbf_sph_; }
    bool is_spherical() const { return is_spherical_; }
    bool needs_transformation() const { return needs_transform_; }
    
    // Akses transformer
    const SphericalTransformer& get_transformer() const { return transformer_; }
    const BasisTransformationHelper& get_helper() const { return *helper_; }
    
    void debug_print_offsets() const;


    
private:
    const BasisSet& basis_;
    const Molecule& mol_;  // TAMBAHKAN ini
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

/**
 * @brief Helper untuk UHF dengan basis spherical
 */
class UHFSphericalHelper {
public:
    UHFSphericalHelper(
        const BasisSet& basis,
        const Molecule& mol,
        bool use_spherical = true
    );
    
    // Prepare integrals untuk UHF
    struct UHFIntegrals {
        Eigen::MatrixXd S;  // Overlap
        Eigen::MatrixXd T;  // Kinetic
        Eigen::MatrixXd V;  // Nuclear
        Eigen::Tensor<double, 4> ERI;  // UBAH dari std::vector<double>
        int nbf;
        bool is_spherical;
    };
    
    UHFIntegrals prepare_integrals();
    
    // Transform hasil UHF kembali jika diperlukan
    Eigen::MatrixXd transform_density_to_cartesian(const Eigen::MatrixXd& D_sph);
    Eigen::MatrixXd transform_mo_to_cartesian(const Eigen::MatrixXd& C_sph);
    
private:
    SphericalIntegralEngine engine_;
    const BasisSet& basis_;
    const Molecule& mol_;
    bool use_spherical_;
};

/**
 * @brief Utility functions untuk spherical basis
 */
namespace spherical_utils {

// Check apakah basis memerlukan spherical
bool requires_spherical_transformation(const BasisSet& basis);

// Print info transformasi
void print_transformation_info(
    const BasisTransformationHelper& helper,
    std::ostream& os = std::cout
);

// Validate transformasi (debugging)
bool validate_transformation(
    const Eigen::MatrixXd& cart_matrix,
    const Eigen::MatrixXd& sph_matrix,
    const SphericalTransformer& transformer,
    const std::vector<int>& angular_momenta,
    double threshold = 1e-10
);

// Compute memory requirement
size_t estimate_memory_requirement(
    int nbf_cart,
    int nbf_sph,
    bool store_eri = true
);

} // namespace spherical_utils

/**
 * @brief Wrapper untuk transformasi quick access
 */
class QuickSphericalTransform {
public:
    // Static functions untuk transformasi cepat
    static Eigen::MatrixXd transform_matrix(
        const Eigen::MatrixXd& cart_matrix,
        const BasisSet& basis,
        const Molecule& mol  // TAMBAHKAN parameter
    );
    
    static Eigen::Tensor<double, 4> transform_eri(  // UBAH return type
        const Eigen::Tensor<double, 4>& cart_eri,  // UBAH parameter type
        const BasisSet& basis,
        const Molecule& mol  // TAMBAHKAN parameter
    );
    
    static Eigen::MatrixXd transform_mo_coefficients(
        const Eigen::MatrixXd& cart_coeff,
        const BasisSet& basis,
        const Molecule& mol  // TAMBAHKAN parameter
    );
    
private:
    QuickSphericalTransform() = delete;
};

} // namespace mshqc

#endif // MSHQC_SPHERICAL_INTEGRATION_H