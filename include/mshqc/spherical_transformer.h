// spherical_transformer.h
// Author: Muhamad Syahrul Hidayat
// Modul untuk transformasi Cartesian ke Spherical harmonics

#ifndef MSHQC_SPHERICAL_TRANSFORMER_H
#define MSHQC_SPHERICAL_TRANSFORMER_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <map>

namespace mshqc {

/**
 * @brief Kelas untuk transformasi basis Cartesian ke Spherical
 * 
 * Menangani transformasi integral dan koefisien orbital dari
 * basis Cartesian (6d, 10f, dll) ke Spherical (5d, 7f, dll)
 */
class SphericalTransformer {
public:
    SphericalTransformer();
    ~SphericalTransformer() = default;

    // Transformasi matriks 1-elektron (overlap, kinetic, nuclear)
    Eigen::MatrixXd transform_1e_matrix(
        const Eigen::MatrixXd& cart_matrix,
        const std::vector<int>& angular_momenta,
        const std::vector<int>& shell_offsets_cart,
        const std::vector<int>& shell_offsets_sph
    ) const;

    // Transformasi integral 2-elektron (ERI)
    std::vector<double> transform_2e_integrals(
        const std::vector<double>& cart_eris,
        const std::vector<int>& angular_momenta,
        int nbf_cart,
        int nbf_sph
    );

    // Transformasi koefisien orbital
    Eigen::MatrixXd transform_mo_coefficients(
        const Eigen::MatrixXd& cart_coeff,
        const std::vector<int>& angular_momenta,
        const std::vector<int>& shell_offsets_cart,
        const std::vector<int>& shell_offsets_sph
    ) const;

    // Mendapatkan matriks transformasi untuk angular momentum tertentu
    Eigen::MatrixXd get_transformation_matrix(int l)const;

    // Mendapatkan ukuran basis
    int get_cartesian_size(int l) const;
    int get_spherical_size(int l) const;

    // Utilitas
    bool is_spherical_basis(const std::vector<int>& angular_momenta) const;
    int count_spherical_functions(const std::vector<int>& angular_momenta) const;
    int count_cartesian_functions(const std::vector<int>& angular_momenta) const;

    // Helper untuk transformasi 2-elektron
    void transform_eri_shell_quartet(
        const double* cart_eri,
        double* sph_eri,
        const Eigen::MatrixXd& T1,
        const Eigen::MatrixXd& T2,
        const Eigen::MatrixXd& T3,
        const Eigen::MatrixXd& T4,
        int n1_cart, int n2_cart, int n3_cart, int n4_cart,
        int n1_sph, int n2_sph, int n3_sph, int n4_sph
    );


private:
    // Matriks transformasi untuk setiap angular momentum
    std::map<int, Eigen::MatrixXd> transformation_matrices_;

    // Inisialisasi matriks transformasi
    void initialize_transformation_matrices();
    
    // Matriks transformasi spesifik
    Eigen::MatrixXd get_s_transform();  // l=0
    Eigen::MatrixXd get_p_transform();  // l=1
    Eigen::MatrixXd get_d_transform();  // l=2
    Eigen::MatrixXd get_f_transform();  // l=3
    Eigen::MatrixXd get_g_transform();  // l=4

    
    // Ordering functions
    int cartesian_index(int l, int i, int j, int k) const;
    int spherical_index(int l, int m) const;
};

/**
 * @brief Helper class untuk manajemen basis transformasi
 */
class BasisTransformationHelper {
public:
    struct ShellInfo {
        int angular_momentum;
        int cart_offset;
        int sph_offset;
        int cart_size;
        int sph_size;
    };
// ==================== Helper Functions (Deklarasi) ====================
// Fungsi helper untuk menghitung offset shell
std::vector<int> compute_shell_offsets_cartesian(
    const std::vector<int>& angular_momenta
);

std::vector<int> compute_shell_offsets_spherical(
    const std::vector<int>& angular_momenta
);    

    BasisTransformationHelper(const std::vector<int>& angular_momenta);

    const std::vector<ShellInfo>& get_shell_info() const { return shells_; }
    int get_total_cart_functions() const { return total_cart_; }
    int get_total_sph_functions() const { return total_sph_; }
    bool needs_transformation() const { return needs_transform_; }

private:
    std::vector<ShellInfo> shells_;
    int total_cart_;
    int total_sph_;
    bool needs_transform_;
};
// ==================== Helper Functions (Deklarasi) ====================
// Fungsi helper untuk menghitung offset shell
std::vector<int> compute_shell_offsets_cartesian(
    const std::vector<int>& angular_momenta
);

std::vector<int> compute_shell_offsets_spherical(
    const std::vector<int>& angular_momenta
);    

} // namespace mshqc

#endif // MSHQC_SPHERICAL_TRANSFORMER_H