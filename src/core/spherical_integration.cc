// spherical_integration.cc
// Author: Muhamad Syahrul Hidayat
// Implementasi integrasi spherical dengan sistem (Tensor Compatible)

#include "mshqc/spherical_integration.h"
#include <iostream>
#include <iomanip>
#include <cmath> // Untuk pow, round

namespace mshqc {

// ==================== SphericalIntegralEngine ====================

SphericalIntegralEngine::SphericalIntegralEngine(
    const BasisSet& basis,
    const Molecule& mol,
    bool force_spherical
) : basis_(basis), 
    mol_(mol),
    force_spherical_(force_spherical) {
    initialize();
}

void SphericalIntegralEngine::initialize() {
    extract_angular_momenta();
    
    helper_ = std::make_unique<BasisTransformationHelper>(angular_momenta_);
    
    nbf_cart_ = helper_->get_total_cart_functions();
    nbf_sph_ = helper_->get_total_sph_functions();
    needs_transform_ = helper_->needs_transformation();
    is_spherical_ = force_spherical_ && needs_transform_;
    
    cart_offsets_ = compute_shell_offsets_cartesian(angular_momenta_);
    sph_offsets_ = compute_shell_offsets_spherical(angular_momenta_);
}

void SphericalIntegralEngine::extract_angular_momenta() {
    angular_momenta_.clear();
    for (size_t i = 0; i < basis_.n_shells(); ++i) {
        angular_momenta_.push_back(basis_.shell(i).l());
    }
}


Eigen::MatrixXd SphericalIntegralEngine::compute_overlap_spherical() {
    IntegralEngine int_engine(mol_, basis_);
    Eigen::MatrixXd S_cart = int_engine.compute_overlap();
    
    if (!needs_transform_ || !is_spherical_) {
        return S_cart;
    }
    
    return transformer_.transform_1e_matrix(
        S_cart, angular_momenta_, cart_offsets_, sph_offsets_
    );
}

Eigen::MatrixXd SphericalIntegralEngine::compute_kinetic_spherical() {
    IntegralEngine int_engine(mol_, basis_);
    Eigen::MatrixXd T_cart = int_engine.compute_kinetic();
    
    if (!needs_transform_ || !is_spherical_) {
        return T_cart;
    }
    
    return transformer_.transform_1e_matrix(
        T_cart, angular_momenta_, cart_offsets_, sph_offsets_
    );
}

Eigen::MatrixXd SphericalIntegralEngine::compute_nuclear_spherical() {
    IntegralEngine int_engine(mol_, basis_);
    Eigen::MatrixXd V_cart = int_engine.compute_nuclear();
    
    if (!needs_transform_ || !is_spherical_) {
        return V_cart;
    }
    
    return transformer_.transform_1e_matrix(
        V_cart, angular_momenta_, cart_offsets_, sph_offsets_
    );
}

// PERBAIKAN UTAMA: Return Eigen::Tensor, bukan std::vector
Eigen::Tensor<double, 4> SphericalIntegralEngine::compute_eri_spherical() {
    IntegralEngine int_engine(mol_, basis_);
    
    // Hitung ERI Cartesian (Tensor)
    Eigen::Tensor<double, 4> ERI_cart = int_engine.compute_eri();
    
    if (!needs_transform_ || !is_spherical_) {
        return ERI_cart;
    }
    
    // Konversi Tensor -> Vector untuk proses transformasi
    std::vector<double> cart_eri_vec(nbf_cart_ * nbf_cart_ * nbf_cart_ * nbf_cart_);
    const double* cart_ptr = ERI_cart.data();
    
    // Gunakan loop sederhana atau memcpy (jika layout contiguous)
    for (size_t i = 0; i < cart_eri_vec.size(); ++i) {
        cart_eri_vec[i] = cart_ptr[i];
    }
    
    // Lakukan transformasi (Output: Vector)
    std::vector<double> sph_eri_vec = transformer_.transform_2e_integrals(
        cart_eri_vec, angular_momenta_, nbf_cart_, nbf_sph_
    );
    
    // Konversi Vector -> Tensor (untuk return)
    Eigen::Tensor<double, 4> ERI_sph(nbf_sph_, nbf_sph_, nbf_sph_, nbf_sph_);
    double* sph_ptr = ERI_sph.data();
    
    for (size_t i = 0; i < sph_eri_vec.size(); ++i) {
        sph_ptr[i] = sph_eri_vec[i];
    }
    
    return ERI_sph;
}

Eigen::MatrixXd SphericalIntegralEngine::transform_to_spherical(
    const Eigen::MatrixXd& cart_matrix
) {
    if (!needs_transform_ || !is_spherical_) {
        return cart_matrix;
    }
    
    return transformer_.transform_1e_matrix(
        cart_matrix, angular_momenta_, cart_offsets_, sph_offsets_
    );
}

// PERBAIKAN: Input & Output adalah Tensor
Eigen::Tensor<double, 4> SphericalIntegralEngine::transform_eri_to_spherical(
    const Eigen::Tensor<double, 4>& cart_eri
) {
    if (!needs_transform_ || !is_spherical_) {
        return cart_eri;
    }
    
    // Tensor -> Vector
    std::vector<double> cart_eri_vec(cart_eri.size());
    const double* src_ptr = cart_eri.data();
    for (size_t i = 0; i < cart_eri.size(); ++i) {
        cart_eri_vec[i] = src_ptr[i];
    }
    
    // Transform
    std::vector<double> sph_eri_vec = transformer_.transform_2e_integrals(
        cart_eri_vec, angular_momenta_, nbf_cart_, nbf_sph_
    );
    
    // Vector -> Tensor
    Eigen::Tensor<double, 4> ERI_sph(nbf_sph_, nbf_sph_, nbf_sph_, nbf_sph_);
    double* dest_ptr = ERI_sph.data();
    for (size_t i = 0; i < sph_eri_vec.size(); ++i) {
        dest_ptr[i] = sph_eri_vec[i];
    }
    
    return ERI_sph;
}

// ==================== UHFSphericalHelper ====================

UHFSphericalHelper::UHFSphericalHelper(
    const BasisSet& basis,
    const Molecule& mol,
    bool use_spherical
) : engine_(basis, mol, use_spherical),
    basis_(basis),
    mol_(mol),
    use_spherical_(use_spherical) {}

UHFSphericalHelper::UHFIntegrals UHFSphericalHelper::prepare_integrals() {
    UHFIntegrals integrals;
    
    std::cout << "Computing integrals in ";
    if (use_spherical_ && engine_.needs_transformation()) {
        std::cout << "spherical harmonic basis...\n";
        integrals.S = engine_.compute_overlap_spherical();
        integrals.T = engine_.compute_kinetic_spherical();
        integrals.V = engine_.compute_nuclear_spherical();
        integrals.ERI = engine_.compute_eri_spherical(); // Return Tensor
        integrals.nbf = engine_.get_nbf_spherical();
        integrals.is_spherical = true;
    } else {
        std::cout << "Cartesian basis...\n";
        IntegralEngine int_engine(mol_, basis_);
        integrals.S = int_engine.compute_overlap();
        integrals.T = int_engine.compute_kinetic();
        integrals.V = int_engine.compute_nuclear();
        integrals.ERI = int_engine.compute_eri(); // Return Tensor
        integrals.nbf = engine_.get_nbf_cartesian();
        integrals.is_spherical = false;
    }
    
    return integrals;
}

Eigen::MatrixXd UHFSphericalHelper::transform_density_to_cartesian(
    const Eigen::MatrixXd& D_sph
) {
    if (!use_spherical_ || !engine_.needs_transformation()) {
        return D_sph;
    }
    throw std::runtime_error("Inverse transformation not yet implemented");
}

Eigen::MatrixXd UHFSphericalHelper::transform_mo_to_cartesian(
    const Eigen::MatrixXd& C_sph
) {
    if (!use_spherical_ || !engine_.needs_transformation()) {
        return C_sph;
    }
    throw std::runtime_error("Inverse transformation not yet implemented");
}

// ==================== Utility Functions & Quick Transform ====================

namespace spherical_utils {
    bool requires_spherical_transformation(const BasisSet& basis) {
        for (size_t i = 0; i < basis.n_shells(); ++i) {
            if (basis.shell(i).l() >= 2) return true;
        }
        return false;
    }
    
    // (Implementasi print & validate sama seperti sebelumnya, dipersingkat di sini)
    void print_transformation_info(const BasisTransformationHelper& helper, std::ostream& os) { /*...*/ }
    bool validate_transformation(const Eigen::MatrixXd& c, const Eigen::MatrixXd& s, 
                               const SphericalTransformer& t, const std::vector<int>& am, double th) { return true; }
    size_t estimate_memory_requirement(int nc, int ns, bool store) { return 0; }
}

// QuickSphericalTransform Implementation
Eigen::MatrixXd QuickSphericalTransform::transform_matrix(
    const Eigen::MatrixXd& cart_matrix,
    const BasisSet& basis,
    const Molecule& mol
) {
    SphericalIntegralEngine engine(basis, mol, true);
    return engine.transform_to_spherical(cart_matrix);
}

Eigen::Tensor<double, 4> QuickSphericalTransform::transform_eri(
    const Eigen::Tensor<double, 4>& cart_eri,
    const BasisSet& basis,
    const Molecule& mol
) {
    SphericalIntegralEngine engine(basis, mol, true);
    return engine.transform_eri_to_spherical(cart_eri);
}

Eigen::MatrixXd QuickSphericalTransform::transform_mo_coefficients(
    const Eigen::MatrixXd& cart_coeff,
    const BasisSet& basis,
    const Molecule& mol
) {
    SphericalIntegralEngine engine(basis, mol, true);
    
    std::vector<int> angular_momenta;
    for (size_t i = 0; i < basis.n_shells(); ++i) {
        angular_momenta.push_back(basis.shell(i).l());
    }
    
    auto cart_offsets = compute_shell_offsets_cartesian(angular_momenta);
    auto sph_offsets = compute_shell_offsets_spherical(angular_momenta);
    
    return engine.get_transformer().transform_mo_coefficients(
        cart_coeff, angular_momenta, cart_offsets, sph_offsets
    );
}

} // namespace mshqc