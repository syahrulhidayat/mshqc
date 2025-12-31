// spherical_transformer.cc (Part 1)
// Author: Muhamad Syahrul Hidayat
// Implementasi transformasi Cartesian ke Spherical

#include "mshqc/spherical_transformer.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace mshqc {
    


SphericalTransformer::SphericalTransformer() {
    initialize_transformation_matrices();
}

void SphericalTransformer::initialize_transformation_matrices() {
    // Inisialisasi matriks untuk l=0 sampai l=4
    transformation_matrices_[0] = get_s_transform();
    transformation_matrices_[1] = get_p_transform();
    transformation_matrices_[2] = get_d_transform();
    transformation_matrices_[3] = get_f_transform();
    transformation_matrices_[4] = get_g_transform();
}

// ==================== Ukuran Basis ====================
int SphericalTransformer::get_cartesian_size(int l) const {
    return (l + 1) * (l + 2) / 2;
}

int SphericalTransformer::get_spherical_size(int l) const {
    return 2 * l + 1;
}

// ==================== Matriks Transformasi s (l=0) ====================
Eigen::MatrixXd SphericalTransformer::get_s_transform()  {
    // s orbital: 1 Cartesian = 1 Spherical (identitas)
    Eigen::MatrixXd T(1, 1);
    T(0, 0) = 1.0;
    return T;
}

// ==================== Matriks Transformasi p (l=1) ====================
Eigen::MatrixXd SphericalTransformer::get_p_transform() {
    // p orbital: 3 Cartesian (x,y,z) = 3 Spherical (p-1,p0,p+1)
    // Order Cartesian: x, y, z
    // Order Spherical: p-1, p0, p+1
    Eigen::MatrixXd T(3, 3);
    T.setZero();
    
    // p-1 (m=-1): py
    T(0, 1) = 1.0;
    
    // p0 (m=0): pz
    T(1, 2) = 1.0;
    
    // p+1 (m=+1): px
    T(2, 0) = 1.0;
    
    return T;
}

// ==================== Matriks Transformasi d (l=2) ====================
Eigen::MatrixXd SphericalTransformer::get_d_transform() {
    // d orbital: 6 Cartesian → 5 Spherical
    // Cartesian order: xx, yy, zz, xy, xz, yz
    // Spherical order: d-2, d-1, d0, d+1, d+2
    
    Eigen::MatrixXd T(5, 6);
    T.setZero();
    
    const double sqrt3 = std::sqrt(3.0);
    const double sqrt3_inv = 1.0 / sqrt3;
    
    // d-2 (m=-2): dxy → sqrt(3) * xy
    T(0, 3) = sqrt3;
    
    // d-1 (m=-1): dyz → sqrt(3) * yz
    T(1, 5) = sqrt3;
    
    // d0 (m=0): d(3z²-r²) → (-1/2)(xx + yy) + zz
    T(2, 0) = -0.5;  // xx
    T(2, 1) = -0.5;  // yy
    T(2, 2) = 1.0;   // zz
    
    // d+1 (m=+1): dxz → sqrt(3) * xz
    T(3, 4) = sqrt3;
    
    // d+2 (m=+2): d(x²-y²) → (sqrt(3)/2)(xx - yy)
    T(4, 0) = sqrt3 / 2.0;   // xx
    T(4, 1) = -sqrt3 / 2.0;  // yy
    
    return T;
}

// ==================== Matriks Transformasi f (l=3) ====================
Eigen::MatrixXd SphericalTransformer::get_f_transform() {
    // f orbital: 10 Cartesian → 7 Spherical
    // Cartesian order: xxx, yyy, zzz, xxy, xxz, xyy, yyz, xzz, yzz, xyz
    // Spherical order: f-3, f-2, f-1, f0, f+1, f+2, f+3
    
    Eigen::MatrixXd T(7, 10);
    T.setZero();
    
    const double sqrt5 = std::sqrt(5.0);
    const double sqrt6 = std::sqrt(6.0);
    const double sqrt10 = std::sqrt(10.0);
    const double sqrt15 = std::sqrt(15.0);
    
    // f-3 (m=-3): f(y(3x²-y²)) 
    T(0, 3) = -sqrt10 / 2.0;  // xxy
    T(0, 5) = sqrt10 / 2.0;   // xyy
    
    // f-2 (m=-2): f(xyz)
    T(1, 9) = sqrt15;  // xyz
    
    // f-1 (m=-1): f(y(5z²-r²))
    T(2, 5) = -sqrt6 / 4.0;   // xyy
    T(2, 6) = sqrt6 / 2.0;    // yyz
    T(2, 8) = -sqrt6 / 4.0;   // yzz
    
    // f0 (m=0): f(z(5z²-3r²))
    T(3, 2) = 1.0;            // zzz
    T(3, 4) = -3.0 / (2.0 * sqrt5);  // xxz
    T(3, 6) = -3.0 / (2.0 * sqrt5);  // yyz
    
    // f+1 (m=+1): f(x(5z²-r²))
    T(4, 3) = -sqrt6 / 4.0;   // xxy
    T(4, 4) = sqrt6 / 2.0;    // xxz
    T(4, 7) = -sqrt6 / 4.0;   // xzz
    
    // f+2 (m=+2): f(z(x²-y²))
    T(5, 4) = sqrt15 / 2.0;   // xxz
    T(5, 6) = -sqrt15 / 2.0;  // yyz
    
    // f+3 (m=+3): f(x(x²-3y²))
    T(6, 0) = sqrt10 / 2.0;   // xxx
    T(6, 5) = -sqrt10 / 2.0;  // xyy
    
    return T;
}

// ==================== Matriks Transformasi g (l=4) ====================
Eigen::MatrixXd SphericalTransformer::get_g_transform() {
    // g orbital: 15 Cartesian → 9 Spherical
    // Simplified: untuk basis sangat besar seperti cc-pV5Z
    
    Eigen::MatrixXd T(9, 15);
    T.setZero();
    
    const double sqrt5 = std::sqrt(5.0);
    const double sqrt7 = std::sqrt(7.0);
    const double sqrt35 = std::sqrt(35.0);
    const double sqrt70 = std::sqrt(70.0);
    
    // Cartesian order untuk l=4:
    // xxxx, yyyy, zzzz, xxxy, xxxz, xyyy, yyyz, xzzz, yzzz, 
    // xxyy, xxzz, yyzz, xxyz, xyyz, xyzz
    
    // g-4 (m=-4): simplified
    T(0, 3) = sqrt35 / 2.0;   // xxxy
    T(0, 5) = -sqrt35 / 2.0;  // xyyy
    
    // g-3 (m=-3): simplified
    T(1, 12) = -sqrt70 / 2.0; // xxyz
    T(1, 13) = sqrt70 / 2.0;  // xyyz
    
    // g-2 (m=-2): simplified
    T(2, 9) = -3.0 * sqrt5 / 4.0;   // xxyy
    T(2, 11) = sqrt5 / 2.0;         // yyzz
    T(2, 14) = sqrt5;               // xyzz
    
    // g-1 (m=-1): simplified
    T(3, 13) = -sqrt5 / 4.0;  // xyyz
    T(3, 3) = sqrt5 / 4.0;    // xxxy
    
    // g0 (m=0): main component
    T(4, 2) = 1.0;  // zzzz
    T(4, 10) = -3.0 / sqrt7;  // xxzz
    T(4, 11) = -3.0 / sqrt7;  // yyzz
    T(4, 9) = 3.0 / (4.0 * sqrt7);  // xxyy
    
    // g+1 (m=+1): simplified
    T(5, 12) = -sqrt5 / 4.0;  // xxyz
    T(5, 5) = sqrt5 / 4.0;    // xyyy
    
    // g+2 (m=+2): simplified
    T(6, 9) = 3.0 * sqrt5 / 4.0;  // xxyy
    T(6, 10) = sqrt5 / 2.0;       // xxzz
    T(6, 14) = -sqrt5;            // xyzz
    
    // g+3 (m=+3): simplified
    T(7, 4) = sqrt70 / 2.0;   // xxxz
    T(7, 6) = -sqrt70 / 2.0;  // yyyz
    
    // g+4 (m=+4): simplified
    T(8, 0) = sqrt35 / 2.0;   // xxxx
    T(8, 1) = sqrt35 / 2.0;   // yyyy
    T(8, 9) = -3.0 * sqrt35 / 4.0;  // xxyy
    
    return T;
}

// ==================== Get Transformation Matrix ====================
Eigen::MatrixXd SphericalTransformer::get_transformation_matrix(int l) const {
    auto it = transformation_matrices_.find(l);
    if (it != transformation_matrices_.end()) {
        return it->second;
    }
    
    throw std::runtime_error("Transformation matrix for l=" + 
                           std::to_string(l) + " not implemented");
}

// spherical_transformer.cc (Part 2)
// Author: Muhamad Syahrul Hidayat
// Transformasi 1-elektron dan MO coefficients


// ==================== Transformasi Matriks 1-Elektron ====================
Eigen::MatrixXd SphericalTransformer::transform_1e_matrix(
    const Eigen::MatrixXd& cart_matrix,
    const std::vector<int>& angular_momenta,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph
)const {
    int nshells = angular_momenta.size();
    int nbf_cart = cart_matrix.rows();
    int nbf_sph = shell_offsets_sph.back();
    
    Eigen::MatrixXd sph_matrix = Eigen::MatrixXd::Zero(nbf_sph, nbf_sph);
    
    // Loop over shell pairs
    for (int ishell = 0; ishell < nshells; ++ishell) {
        int li = angular_momenta[ishell];
        int cart_offset_i = shell_offsets_cart[ishell];
        int sph_offset_i = shell_offsets_sph[ishell];
        int ncart_i = get_cartesian_size(li);
        int nsph_i = get_spherical_size(li);
        
        Eigen::MatrixXd Ti = get_transformation_matrix(li);
        
        for (int jshell = 0; jshell < nshells; ++jshell) {
            int lj = angular_momenta[jshell];
            int cart_offset_j = shell_offsets_cart[jshell];
            int sph_offset_j = shell_offsets_sph[jshell];
            int ncart_j = get_cartesian_size(lj);
            int nsph_j = get_spherical_size(lj);
            
            Eigen::MatrixXd Tj = get_transformation_matrix(lj);
            
            // Extract Cartesian block
            Eigen::MatrixXd cart_block = cart_matrix.block(
                cart_offset_i, cart_offset_j, ncart_i, ncart_j
            );
            
            // Transform: S_sph = T_i * S_cart * T_j^T
            Eigen::MatrixXd sph_block = Ti * cart_block * Tj.transpose();
            
            // Place in spherical matrix
            sph_matrix.block(sph_offset_i, sph_offset_j, nsph_i, nsph_j) = sph_block;
        }
    }
    
    return sph_matrix;
}

// ==================== Transformasi MO Coefficients ====================
Eigen::MatrixXd SphericalTransformer::transform_mo_coefficients(
    const Eigen::MatrixXd& cart_coeff,
    const std::vector<int>& angular_momenta,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph
) const {
    int nshells = angular_momenta.size();
    int nbf_cart = cart_coeff.rows();
    int nmo = cart_coeff.cols();
    int nbf_sph = shell_offsets_sph.back();
    
    Eigen::MatrixXd sph_coeff = Eigen::MatrixXd::Zero(nbf_sph, nmo);
    
    // Loop over shells
    for (int ishell = 0; ishell < nshells; ++ishell) {
        int li = angular_momenta[ishell];
        int cart_offset = shell_offsets_cart[ishell];
        int sph_offset = shell_offsets_sph[ishell];
        int ncart = get_cartesian_size(li);
        int nsph = get_spherical_size(li);
        
        Eigen::MatrixXd T = get_transformation_matrix(li);
        
        // Extract Cartesian block for all MOs
        Eigen::MatrixXd cart_block = cart_coeff.block(cart_offset, 0, ncart, nmo);
        
        // Transform: C_sph = T * C_cart
        Eigen::MatrixXd sph_block = T * cart_block;
        
        // Place in spherical coefficient matrix
        sph_coeff.block(sph_offset, 0, nsph, nmo) = sph_block;
    }
    
    return sph_coeff;
}

// ==================== Utilitas ====================
bool SphericalTransformer::is_spherical_basis(
    const std::vector<int>& angular_momenta
) const {
    // Basis dianggap spherical jika ada l >= 2
    for (int l : angular_momenta) {
        if (l >= 2) {
            return true;
        }
    }
    return false;
}

int SphericalTransformer::count_spherical_functions(
    const std::vector<int>& angular_momenta
) const {
    int count = 0;
    for (int l : angular_momenta) {
        count += get_spherical_size(l);
    }
    return count;
}

int SphericalTransformer::count_cartesian_functions(
    const std::vector<int>& angular_momenta
) const {
    int count = 0;
    for (int l : angular_momenta) {
        count += get_cartesian_size(l);
    }
    return count;
}

// ==================== Indexing Functions ====================
int SphericalTransformer::cartesian_index(int l, int i, int j, int k) const {
    // Lexicographic ordering: x^i y^j z^k where i+j+k = l
    // Example for l=2: xx(2,0,0), yy(0,2,0), zz(0,0,2), 
    //                   xy(1,1,0), xz(1,0,1), yz(0,1,1)
    
    if (i + j + k != l) {
        throw std::runtime_error("Invalid Cartesian index: i+j+k != l");
    }
    
    int idx = 0;
    for (int ii = l; ii >= 0; --ii) {
        for (int jj = l - ii; jj >= 0; --jj) {
            int kk = l - ii - jj;
            if (ii == i && jj == j && kk == k) {
                return idx;
            }
            idx++;
        }
    }
    
    return -1; // Should never reach here
}

int SphericalTransformer::spherical_index(int l, int m) const {
    // Spherical index: m ranges from -l to +l
    // Index = m + l (so index goes from 0 to 2l)
    
    if (std::abs(m) > l) {
        throw std::runtime_error("Invalid spherical index: |m| > l");
    }
    
    return m + l;
}

// ==================== BasisTransformationHelper ====================
BasisTransformationHelper::BasisTransformationHelper(
    const std::vector<int>& angular_momenta
) : total_cart_(0), total_sph_(0), needs_transform_(false) {
    
    SphericalTransformer transformer;
    
    for (size_t i = 0; i < angular_momenta.size(); ++i) {
        int l = angular_momenta[i];
        
        ShellInfo shell;
        shell.angular_momentum = l;
        shell.cart_offset = total_cart_;
        shell.sph_offset = total_sph_;
        shell.cart_size = transformer.get_cartesian_size(l);
        shell.sph_size = transformer.get_spherical_size(l);
        
        shells_.push_back(shell);
        
        total_cart_ += shell.cart_size;
        total_sph_ += shell.sph_size;
        
        // Check if transformation needed (l >= 2)
        if (l >= 2 && shell.cart_size != shell.sph_size) {
            needs_transform_ = true;
        }
    }
}

// ==================== Helper Functions untuk Offset ====================
std::vector<int> compute_shell_offsets_cartesian(
    const std::vector<int>& angular_momenta
) {
    std::vector<int> offsets;
    offsets.reserve(angular_momenta.size() + 1);
    
    SphericalTransformer transformer;
    int offset = 0;
    
    for (int l : angular_momenta) {
        offsets.push_back(offset);
        offset += transformer.get_cartesian_size(l);
    }
    offsets.push_back(offset); // Total size
    
    return offsets;
}

std::vector<int> compute_shell_offsets_spherical(
    const std::vector<int>& angular_momenta
) {
    std::vector<int> offsets;
    offsets.reserve(angular_momenta.size() + 1);
    
    SphericalTransformer transformer;
    int offset = 0;
    
    for (int l : angular_momenta) {
        offsets.push_back(offset);
        offset += transformer.get_spherical_size(l);
    }
    offsets.push_back(offset); // Total size
    
    return offsets;
}

// spherical_transformer.cc (Part 3)
// Author: Muhamad Syahrul Hidayat
// Transformasi 2-elektron integrals (ERI)



// ==================== Transformasi 2-Elektron Integrals ====================
std::vector<double> SphericalTransformer::transform_2e_integrals(
    const std::vector<double>& cart_eris,
    const std::vector<int>& angular_momenta,
    int nbf_cart,
    int nbf_sph
) {
    // Alokasi memori untuk spherical ERIs
    size_t n_sph_integrals = static_cast<size_t>(nbf_sph) * nbf_sph * 
                             nbf_sph * nbf_sph;
    std::vector<double> sph_eris(n_sph_integrals, 0.0);
    
    // Compute shell offsets
    std::vector<int> cart_offsets = compute_shell_offsets_cartesian(angular_momenta);
    std::vector<int> sph_offsets = compute_shell_offsets_spherical(angular_momenta);
    
    int nshells = angular_momenta.size();
    
    // Loop over shell quartets (i,j,k,l)
    for (int ishell = 0; ishell < nshells; ++ishell) {
        int li = angular_momenta[ishell];
        int cart_i = cart_offsets[ishell];
        int sph_i = sph_offsets[ishell];
        int ncart_i = get_cartesian_size(li);
        int nsph_i = get_spherical_size(li);
        Eigen::MatrixXd Ti = get_transformation_matrix(li);
        
        for (int jshell = 0; jshell < nshells; ++jshell) {
            int lj = angular_momenta[jshell];
            int cart_j = cart_offsets[jshell];
            int sph_j = sph_offsets[jshell];
            int ncart_j = get_cartesian_size(lj);
            int nsph_j = get_spherical_size(lj);
            Eigen::MatrixXd Tj = get_transformation_matrix(lj);
            
            for (int kshell = 0; kshell < nshells; ++kshell) {
                int lk = angular_momenta[kshell];
                int cart_k = cart_offsets[kshell];
                int sph_k = sph_offsets[kshell];
                int ncart_k = get_cartesian_size(lk);
                int nsph_k = get_spherical_size(lk);
                Eigen::MatrixXd Tk = get_transformation_matrix(lk);
                
                for (int lshell = 0; lshell < nshells; ++lshell) {
                    int ll = angular_momenta[lshell];
                    int cart_l = cart_offsets[lshell];
                    int sph_l = sph_offsets[lshell];
                    int ncart_l = get_cartesian_size(ll);
                    int nsph_l = get_spherical_size(ll);
                    Eigen::MatrixXd Tl = get_transformation_matrix(ll);
                    
                    // Extract Cartesian ERI block
                    size_t cart_size = ncart_i * ncart_j * ncart_k * ncart_l;
                    std::vector<double> cart_block(cart_size);
                    
                    for (int i = 0; i < ncart_i; ++i) {
                        for (int j = 0; j < ncart_j; ++j) {
                            for (int k = 0; k < ncart_k; ++k) {
                                for (int l = 0; l < ncart_l; ++l) {
                                    size_t cart_idx = 
                                        ((cart_i + i) * nbf_cart + (cart_j + j)) * 
                                        nbf_cart * nbf_cart +
                                        (cart_k + k) * nbf_cart + (cart_l + l);
                                    
                                    size_t block_idx = 
                                        ((i * ncart_j + j) * ncart_k + k) * 
                                        ncart_l + l;
                                    
                                    cart_block[block_idx] = cart_eris[cart_idx];
                                }
                            }
                        }
                    }
                    
                    // Transform shell quartet
                    std::vector<double> sph_block(nsph_i * nsph_j * nsph_k * nsph_l);
                    transform_eri_shell_quartet(
                        cart_block.data(),
                        sph_block.data(),
                        Ti, Tj, Tk, Tl,
                        ncart_i, ncart_j, ncart_k, ncart_l,
                        nsph_i, nsph_j, nsph_k, nsph_l
                    );
                    
                    // Place in spherical ERI array
                    for (int i = 0; i < nsph_i; ++i) {
                        for (int j = 0; j < nsph_j; ++j) {
                            for (int k = 0; k < nsph_k; ++k) {
                                for (int l = 0; l < nsph_l; ++l) {
                                    size_t sph_idx = 
                                        ((sph_i + i) * nbf_sph + (sph_j + j)) * 
                                        nbf_sph * nbf_sph +
                                        (sph_k + k) * nbf_sph + (sph_l + l);
                                    
                                    size_t block_idx = 
                                        ((i * nsph_j + j) * nsph_k + k) * 
                                        nsph_l + l;
                                    
                                    sph_eris[sph_idx] = sph_block[block_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return sph_eris;
}


// ==================== Transform ERI Shell Quartet ====================
void SphericalTransformer::transform_eri_shell_quartet(
    const double* cart_eri,
    double* sph_eri,
    const Eigen::MatrixXd& T1,
    const Eigen::MatrixXd& T2,
    const Eigen::MatrixXd& T3,
    const Eigen::MatrixXd& T4,
    int n1_cart, int n2_cart, int n3_cart, int n4_cart,
    int n1_sph, int n2_sph, int n3_sph, int n4_sph
) {
    // ERI transformation: (μν|λσ)_sph = Σ T1_μi T2_νj T3_λk T4_σl (ij|kl)_cart
    // Implementasi efisien menggunakan transformasi bertahap
    
    // Temporary buffers untuk transformasi bertahap
    std::vector<double> temp1(n1_sph * n2_cart * n3_cart * n4_cart);
    std::vector<double> temp2(n1_sph * n2_sph * n3_cart * n4_cart);
    std::vector<double> temp3(n1_sph * n2_sph * n3_sph * n4_cart);
    
    // Step 1: Transform index 1 (i)
    for (int j = 0; j < n2_cart; ++j) {
        for (int k = 0; k < n3_cart; ++k) {
            for (int l = 0; l < n4_cart; ++l) {
                for (int mu = 0; mu < n1_sph; ++mu) {
                    double sum = 0.0;
                    for (int i = 0; i < n1_cart; ++i) {
                        size_t cart_idx = ((i * n2_cart + j) * n3_cart + k) * n4_cart + l;
                        sum += T1(mu, i) * cart_eri[cart_idx];
                    }
                    size_t temp_idx = ((mu * n2_cart + j) * n3_cart + k) * n4_cart + l;
                    temp1[temp_idx] = sum;
                }
            }
        }
    }
    
    // Step 2: Transform index 2 (j)
    for (int mu = 0; mu < n1_sph; ++mu) {
        for (int k = 0; k < n3_cart; ++k) {
            for (int l = 0; l < n4_cart; ++l) {
                for (int nu = 0; nu < n2_sph; ++nu) {
                    double sum = 0.0;
                    for (int j = 0; j < n2_cart; ++j) {
                        size_t temp1_idx = ((mu * n2_cart + j) * n3_cart + k) * n4_cart + l;
                        sum += T2(nu, j) * temp1[temp1_idx];
                    }
                    size_t temp2_idx = ((mu * n2_sph + nu) * n3_cart + k) * n4_cart + l;
                    temp2[temp2_idx] = sum;
                }
            }
        }
    }
    
    // Step 3: Transform index 3 (k)
    for (int mu = 0; mu < n1_sph; ++mu) {
        for (int nu = 0; nu < n2_sph; ++nu) {
            for (int l = 0; l < n4_cart; ++l) {
                for (int lambda = 0; lambda < n3_sph; ++lambda) {
                    double sum = 0.0;
                    for (int k = 0; k < n3_cart; ++k) {
                        size_t temp2_idx = ((mu * n2_sph + nu) * n3_cart + k) * n4_cart + l;
                        sum += T3(lambda, k) * temp2[temp2_idx];
                    }
                    size_t temp3_idx = ((mu * n2_sph + nu) * n3_sph + lambda) * n4_cart + l;
                    temp3[temp3_idx] = sum;
                }
            }
        }
    }
    
    // Step 4: Transform index 4 (l)
    for (int mu = 0; mu < n1_sph; ++mu) {
        for (int nu = 0; nu < n2_sph; ++nu) {
            for (int lambda = 0; lambda < n3_sph; ++lambda) {
                for (int sigma = 0; sigma < n4_sph; ++sigma) {
                    double sum = 0.0;
                    for (int l = 0; l < n4_cart; ++l) {
                        size_t temp3_idx = ((mu * n2_sph + nu) * n3_sph + lambda) * n4_cart + l;
                        sum += T4(sigma, l) * temp3[temp3_idx];
                    }
                    size_t sph_idx = ((mu * n2_sph + nu) * n3_sph + lambda) * n4_sph + sigma;
                    sph_eri[sph_idx] = sum;
                }
            }
        }
    }
}

// spherical_transformer.cc (Part 4)
// Author: Muhamad Syahrul Hidayat
// Optimasi memory-efficient dan fungsi integrasi



// ==================== Transformasi ERI On-The-Fly (Memory Efficient) ====================
/**
 * @brief Transformasi ERI tanpa alokasi memori penuh
 * Untuk sistem besar, transformasi dilakukan per-batch
 */
class ERITransformerBatch {
public:
    ERITransformerBatch(
        const std::vector<int>& angular_momenta,
        int nbf_cart,
        int nbf_sph,
        size_t batch_size = 10000
    ) : angular_momenta_(angular_momenta),
        nbf_cart_(nbf_cart),
        nbf_sph_(nbf_sph),
        batch_size_(batch_size) {
        
        cart_offsets_ = compute_shell_offsets_cartesian(angular_momenta);
        sph_offsets_ = compute_shell_offsets_spherical(angular_momenta);
    }
    
    // Transform dengan callback untuk setiap batch
    void transform_with_callback(
        const std::vector<double>& cart_eris,
        std::function<void(int, int, int, int, double)> callback
    ) {
        SphericalTransformer transformer;
        int nshells = angular_momenta_.size();
        
        for (int ishell = 0; ishell < nshells; ++ishell) {
            for (int jshell = 0; jshell < nshells; ++jshell) {
                for (int kshell = 0; kshell < nshells; ++kshell) {
                    for (int lshell = 0; lshell < nshells; ++lshell) {
                        process_shell_quartet(
                            transformer, cart_eris, callback,
                            ishell, jshell, kshell, lshell
                        );
                    }
                }
            }
        }
    }
    
private:
    std::vector<int> angular_momenta_;
    std::vector<int> cart_offsets_;
    std::vector<int> sph_offsets_;
    int nbf_cart_;
    int nbf_sph_;
    size_t batch_size_;
    
    void process_shell_quartet(
        SphericalTransformer& transformer,
        const std::vector<double>& cart_eris,
        std::function<void(int, int, int, int, double)>& callback,
        int ishell, int jshell, int kshell, int lshell
    ) {
        int li = angular_momenta_[ishell];
        int lj = angular_momenta_[jshell];
        int lk = angular_momenta_[kshell];
        int ll = angular_momenta_[lshell];
        
        int cart_i = cart_offsets_[ishell];
        int cart_j = cart_offsets_[jshell];
        int cart_k = cart_offsets_[kshell];
        int cart_l = cart_offsets_[lshell];
        
        int sph_i = sph_offsets_[ishell];
        int sph_j = sph_offsets_[jshell];
        int sph_k = sph_offsets_[kshell];
        int sph_l = sph_offsets_[lshell];
        
        int ncart_i = transformer.get_cartesian_size(li);
        int ncart_j = transformer.get_cartesian_size(lj);
        int ncart_k = transformer.get_cartesian_size(lk);
        int ncart_l = transformer.get_cartesian_size(ll);
        
        int nsph_i = transformer.get_spherical_size(li);
        int nsph_j = transformer.get_spherical_size(lj);
        int nsph_k = transformer.get_spherical_size(lk);
        int nsph_l = transformer.get_spherical_size(ll);
        
        Eigen::MatrixXd Ti = transformer.get_transformation_matrix(li);
        Eigen::MatrixXd Tj = transformer.get_transformation_matrix(lj);
        Eigen::MatrixXd Tk = transformer.get_transformation_matrix(lk);
        Eigen::MatrixXd Tl = transformer.get_transformation_matrix(ll);
        
        // Extract and transform
        size_t cart_size = ncart_i * ncart_j * ncart_k * ncart_l;
        std::vector<double> cart_block(cart_size);
        
        for (int i = 0; i < ncart_i; ++i) {
            for (int j = 0; j < ncart_j; ++j) {
                for (int k = 0; k < ncart_k; ++k) {
                    for (int l = 0; l < ncart_l; ++l) {
                        size_t cart_idx = 
                            ((cart_i + i) * nbf_cart_ + (cart_j + j)) * 
                            nbf_cart_ * nbf_cart_ +
                            (cart_k + k) * nbf_cart_ + (cart_l + l);
                        
                        size_t block_idx = 
                            ((i * ncart_j + j) * ncart_k + k) * ncart_l + l;
                        
                        cart_block[block_idx] = cart_eris[cart_idx];
                    }
                }
            }
        }
        
        std::vector<double> sph_block(nsph_i * nsph_j * nsph_k * nsph_l);
        transformer.transform_eri_shell_quartet(
            cart_block.data(), sph_block.data(),
            Ti, Tj, Tk, Tl,
            ncart_i, ncart_j, ncart_k, ncart_l,
            nsph_i, nsph_j, nsph_k, nsph_l
        );
        
        // Call callback for each integral
        for (int i = 0; i < nsph_i; ++i) {
            for (int j = 0; j < nsph_j; ++j) {
                for (int k = 0; k < nsph_k; ++k) {
                    for (int l = 0; l < nsph_l; ++l) {
                        size_t block_idx = 
                            ((i * nsph_j + j) * nsph_k + k) * nsph_l + l;
                        
                        double value = sph_block[block_idx];
                        if (std::abs(value) > 1e-12) {
                            callback(sph_i + i, sph_j + j, 
                                   sph_k + k, sph_l + l, value);
                        }
                    }
                }
            }
        }
    }
};

// ==================== Transformasi Symmetry-Adapted ====================
/**
 * @brief Memanfaatkan simetri 8-fold untuk ERI
 * (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
 */
void transform_2e_integrals_symmetric(
    const std::vector<double>& cart_eris,
    std::vector<double>& sph_eris,
    const std::vector<int>& angular_momenta,
    int nbf_cart,
    int nbf_sph,
    bool use_8fold_symmetry = true
) {
    SphericalTransformer transformer;
    std::vector<int> cart_offsets = compute_shell_offsets_cartesian(angular_momenta);
    std::vector<int> sph_offsets = compute_shell_offsets_spherical(angular_momenta);
    
    int nshells = angular_momenta.size();
    
    for (int ishell = 0; ishell < nshells; ++ishell) {
        for (int jshell = 0; jshell <= (use_8fold_symmetry ? ishell : nshells-1); ++jshell) {
            for (int kshell = 0; kshell < nshells; ++kshell) {
                int lmax = use_8fold_symmetry ? 
                          (ishell == kshell ? jshell : kshell) : nshells-1;
                          
                for (int lshell = 0; lshell <= lmax; ++lshell) {
                    // Transform shell quartet
                    int li = angular_momenta[ishell];
                    int lj = angular_momenta[jshell];
                    int lk = angular_momenta[kshell];
                    int ll = angular_momenta[lshell];
                    
                    // Get transformation matrices
                    Eigen::MatrixXd Ti = transformer.get_transformation_matrix(li);
                    Eigen::MatrixXd Tj = transformer.get_transformation_matrix(lj);
                    Eigen::MatrixXd Tk = transformer.get_transformation_matrix(lk);
                    Eigen::MatrixXd Tl = transformer.get_transformation_matrix(ll);
                    
                    int cart_i = cart_offsets[ishell];
                    int cart_j = cart_offsets[jshell];
                    int cart_k = cart_offsets[kshell];
                    int cart_l = cart_offsets[lshell];
                    
                    int sph_i = sph_offsets[ishell];
                    int sph_j = sph_offsets[jshell];
                    int sph_k = sph_offsets[kshell];
                    int sph_l = sph_offsets[lshell];
                    
                    int ncart_i = transformer.get_cartesian_size(li);
                    int ncart_j = transformer.get_cartesian_size(lj);
                    int ncart_k = transformer.get_cartesian_size(lk);
                    int ncart_l = transformer.get_cartesian_size(ll);
                    
                    int nsph_i = transformer.get_spherical_size(li);
                    int nsph_j = transformer.get_spherical_size(lj);
                    int nsph_k = transformer.get_spherical_size(lk);
                    int nsph_l = transformer.get_spherical_size(ll);
                    
                    // Extract Cartesian block
                    std::vector<double> cart_block(ncart_i * ncart_j * ncart_k * ncart_l);
                    for (int i = 0; i < ncart_i; ++i) {
                        for (int j = 0; j < ncart_j; ++j) {
                            for (int k = 0; k < ncart_k; ++k) {
                                for (int l = 0; l < ncart_l; ++l) {
                                    size_t idx = ((cart_i+i)*nbf_cart + (cart_j+j))*
                                               nbf_cart*nbf_cart + 
                                               (cart_k+k)*nbf_cart + (cart_l+l);
                                    cart_block[((i*ncart_j+j)*ncart_k+k)*ncart_l+l] = 
                                        cart_eris[idx];
                                }
                            }
                        }
                    }
                    
                    // Transform
                    std::vector<double> sph_block(nsph_i * nsph_j * nsph_k * nsph_l);
                    transformer.transform_eri_shell_quartet(
                        cart_block.data(), sph_block.data(),
                        Ti, Tj, Tk, Tl,
                        ncart_i, ncart_j, ncart_k, ncart_l,
                        nsph_i, nsph_j, nsph_k, nsph_l
                    );
                    
                    // Store with symmetry
                    for (int i = 0; i < nsph_i; ++i) {
                        for (int j = 0; j < nsph_j; ++j) {
                            for (int k = 0; k < nsph_k; ++k) {
                                for (int l = 0; l < nsph_l; ++l) {
                                    double val = sph_block[((i*nsph_j+j)*nsph_k+k)*nsph_l+l];
                                    
                                    // Store all 8 symmetric combinations
                                    auto store = [&](int a, int b, int c, int d) {
                                        size_t idx = ((a*nbf_sph+b)*nbf_sph+c)*nbf_sph+d;
                                        sph_eris[idx] = val;
                                    };
                                    
                                    if (use_8fold_symmetry) {
                                        store(sph_i+i, sph_j+j, sph_k+k, sph_l+l);
                                        store(sph_j+j, sph_i+i, sph_k+k, sph_l+l);
                                        store(sph_i+i, sph_j+j, sph_l+l, sph_k+k);
                                        store(sph_j+j, sph_i+i, sph_l+l, sph_k+k);
                                        store(sph_k+k, sph_l+l, sph_i+i, sph_j+j);
                                        store(sph_l+l, sph_k+k, sph_i+i, sph_j+j);
                                        store(sph_k+k, sph_l+l, sph_j+j, sph_i+i);
                                        store(sph_l+l, sph_k+k, sph_j+j, sph_i+i);
                                    } else {
                                        store(sph_i+i, sph_j+j, sph_k+k, sph_l+l);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


} // namespace mshqc