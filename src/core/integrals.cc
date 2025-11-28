/**
 * @file integrals.cc
 * @brief Integral computation engine using libint2
 * 
 * Implementation of IntegralEngine class as wrapper for libint2 library.
 * Computes overlap, kinetic, nuclear attraction, and electron repulsion integrals.
 * 
 * Theory References:
 *   - S. F. Boys, Proc. R. Soc. London A 200, 542 (1950)
 *     [Gaussian integrals analytical solutions]
 *   - M. Head-Gordon & J. A. Pople, J. Chem. Phys. 89, 5777 (1988)
 *     [Efficient integral evaluation algorithms]
 *   - libint2 documentation (external library, BSD license - compatible with MIT)
 *     [https://github.com/evaleev/libint]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note Wrapper implementation around libint2 library.
 *       libint2 itself is BSD-licensed (compatible with MIT).
 *       Our code: original wrapper design, no libint2 code copied.
 *       We only call libint2 public API functions.
 */

#include "mshqc/integrals.h"
#include "mshqc/libcint_wrapper.h"
#include <libint2.hpp>
#include <libint2/config.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace mshqc {

// ============================================================================
// LibintShellData - wraps libint2::Shell vector
// ============================================================================

struct IntegralEngine::LibintShellData {
    std::vector<libint2::Shell> shells;
};

// ============================================================================
// IntegralEngine Implementation
// ============================================================================

IntegralEngine::IntegralEngine(const Molecule& mol, const BasisSet& basis)
    : mol_(mol), basis_(basis), nbasis_(basis.n_basis_functions()),
      libint_shells_(std::make_unique<LibintShellData>()) {
    
    initialize_libint();
    convert_basis_to_libint();
}

IntegralEngine::~IntegralEngine() {
    finalize_libint();
}

void IntegralEngine::initialize_libint() {
    /**
     * Initialize Libint2 library
     * 
     * This must be called once before using any Libint2 functionality.
     * Sets up internal tables and allocates scratch memory.
     * 
     * REFERENCE:
     * Libint2 documentation: https://github.com/evaleev/libint/wiki
     */
    libint2::initialize();
}

void IntegralEngine::finalize_libint() {
    /**
     * Finalize Libint2 library
     * 
     * Cleanup Libint2 resources. Called automatically in destructor.
     */
    libint2::finalize();
}

void IntegralEngine::convert_basis_to_libint() {
    /**
     * Convert QuantChem basis format to Libint2 format
     * 
     * Libint2 uses libint2::Shell which contains:
     * - Gaussian exponents
     * - Contraction coefficients
     * - Angular momentum
     * - Center position
     * - Whether to use pure (spherical) vs. cartesian
     */
    
    libint_shells_->shells.clear();
    libint_shells_->shells.reserve(basis_.n_shells());
    
    for (size_t i = 0; i < basis_.n_shells(); i++) {
        const auto& qc_shell = basis_.shell(i);
        
        // Get angular momentum
        int l = qc_shell.l();
        
        // Get position
        auto pos = qc_shell.position();
        std::array<double, 3> center = {pos[0], pos[1], pos[2]};
        
        // Collect exponents and coefficients
        std::vector<double> exponents;
        std::vector<double> coefficients;
        
        for (size_t j = 0; j < qc_shell.n_primitives(); j++) {
            const auto& prim = qc_shell.primitive(j);
            exponents.push_back(prim.exponent);
            coefficients.push_back(prim.coefficient);
        }
        
        // Create libint2::Shell
        // Note: Libint2 uses pure (spherical) by default
        bool pure = qc_shell.is_spherical();
        
        // Convert std::vector to libint2::svector (boost::container::small_vector)
        libint2::svector<double> alpha_sv(exponents.begin(), exponents.end());
        libint2::svector<double> coeff_sv(coefficients.begin(), coefficients.end());
        
        // Create contraction
        libint2::Shell::Contraction contr;
        contr.l = l;
        contr.pure = pure;
        contr.coeff = coeff_sv;
        
        // Build shell with automatic normalization
        // 4th parameter (true) embeds normalization into coefficients
        libint2::Shell libint_shell(
            alpha_sv,
            {{contr}},
            center
            // default: true - embed normalization
        );
        
        
        libint_shells_->shells.push_back(std::move(libint_shell));
    }
}

Eigen::MatrixXd IntegralEngine::compute_overlap() {
    /**
     * Compute overlap matrix using Libint2
     * 
     * Uses libint2::Engine with Operator::overlap
     * 
     * IMPLEMENTATION:
     * Loop over shell pairs (μ,ν) and compute overlap integrals.
     * Libint2 returns shell quartet results which we extract to
     * individual basis function pairs.
     */
    
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    // Find max number of primitives per shell
    size_t max_nprim = 0;
    for (const auto& shell : libint_shells_->shells) {
        max_nprim = std::max(max_nprim, shell.nprim());
    }
    
    
    // Create Libint2 engine for overlap integrals
    // Constructor: Engine(Operator, max_nprim, max_l, deriv_order=0)
    // Use much larger max_nprim to allocate sufficient primdata
    libint2::Engine engine(libint2::Operator::overlap,
                          100,  // Large value to ensure enough primdata
                          basis_.max_angular_momentum());
    
    const auto& shells = libint_shells_->shells;
    auto shell2bf = basis_.shell_to_basis_function_map();
    
    // Loop over unique shell pairs
    for (size_t s1 = 0; s1 < shells.size(); s1++) {
        size_t bf1_first = shell2bf[s1];
        size_t n1 = shells[s1].size();
        
        for (size_t s2 = 0; s2 <= s1; s2++) {
            size_t bf2_first = shell2bf[s2];
            size_t n2 = shells[s2].size();
            
            // Compute shell pair integrals
            engine.compute(shells[s1], shells[s2]);
            const auto& buf = engine.results();
            
            // buf[0] points to computed integrals (can be nullptr if screened)
            if (buf[0] == nullptr)
                continue;
            
            // Extract integrals to matrix
            for (size_t f1 = 0; f1 < n1; f1++) {
                for (size_t f2 = 0; f2 < n2; f2++) {
                    size_t bf1 = bf1_first + f1;
                    size_t bf2 = bf2_first + f2;
                    
                    double val = buf[0][f1 * n2 + f2];
                    S(bf1, bf2) = val;
                    if (bf1 != bf2)
                        S(bf2, bf1) = val;  // symmetry
                }
            }
        }
    }
    
    return S;
}

Eigen::MatrixXd IntegralEngine::compute_kinetic() {
    /**
     * Compute kinetic energy matrix using Libint2
     * 
     * Uses libint2::Engine with Operator::kinetic
     * 
     * Kinetic energy operator: T = -½∇²
     */
    
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    // Find max number of primitives per shell
    size_t max_nprim = 0;
    for (const auto& shell : libint_shells_->shells) {
        max_nprim = std::max(max_nprim, shell.nprim());
    }
    
    libint2::Engine engine(libint2::Operator::kinetic,
                          100,  // max_nprim
                          basis_.max_angular_momentum());
    
    const auto& shells = libint_shells_->shells;
    auto shell2bf = basis_.shell_to_basis_function_map();
    
    for (size_t s1 = 0; s1 < shells.size(); s1++) {
        size_t bf1_first = shell2bf[s1];
        size_t n1 = shells[s1].size();
        
        for (size_t s2 = 0; s2 <= s1; s2++) {
            size_t bf2_first = shell2bf[s2];
            size_t n2 = shells[s2].size();
            
            engine.compute(shells[s1], shells[s2]);
            const auto& buf = engine.results();
            
            if (buf[0] == nullptr)
                continue;
            
            for (size_t f1 = 0; f1 < n1; f1++) {
                for (size_t f2 = 0; f2 < n2; f2++) {
                    size_t bf1 = bf1_first + f1;
                    size_t bf2 = bf2_first + f2;
                    
                    double val = buf[0][f1 * n2 + f2];
                    T(bf1, bf2) = val;
                    if (bf1 != bf2)
                        T(bf2, bf1) = val;
                }
            }
        }
    }
    
    return T;
}

Eigen::MatrixXd IntegralEngine::compute_nuclear() {
    /**
     * Compute nuclear attraction matrix using Libint2
     * 
     * Uses libint2::Engine with Operator::nuclear
     * 
     * Nuclear attraction: V_nuc = -Σ_A Z_A / |r - R_A|
     * 
     * Libint2 requires list of point charges (atoms) as input.
     */
    
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    // Prepare atom data for Libint2
    std::vector<std::pair<double, std::array<double, 3>>> atoms;
    for (size_t i = 0; i < mol_.n_atoms(); i++) {
        const auto& atom = mol_.atom(i);
        atoms.push_back({
            static_cast<double>(atom.atomic_number),
            {atom.x, atom.y, atom.z}
        });
    }
    
    // Find max number of primitives per shell
    size_t max_nprim = 0;
    for (const auto& shell : libint_shells_->shells) {
        max_nprim = std::max(max_nprim, shell.nprim());
    }
    
    libint2::Engine engine(libint2::Operator::nuclear,
                          100,  // max_nprim
                          basis_.max_angular_momentum());
    engine.set_params(atoms);  // set nuclear charges and positions
    
    const auto& shells = libint_shells_->shells;
    auto shell2bf = basis_.shell_to_basis_function_map();
    
    for (size_t s1 = 0; s1 < shells.size(); s1++) {
        size_t bf1_first = shell2bf[s1];
        size_t n1 = shells[s1].size();
        
        for (size_t s2 = 0; s2 <= s1; s2++) {
            size_t bf2_first = shell2bf[s2];
            size_t n2 = shells[s2].size();
            
            engine.compute(shells[s1], shells[s2]);
            const auto& buf = engine.results();
            
            if (buf[0] == nullptr)
                continue;
            
            for (size_t f1 = 0; f1 < n1; f1++) {
                for (size_t f2 = 0; f2 < n2; f2++) {
                    size_t bf1 = bf1_first + f1;
                    size_t bf2 = bf2_first + f2;
                    
                    double val = buf[0][f1 * n2 + f2];
                    V(bf1, bf2) = val;
                    if (bf1 != bf2)
                        V(bf2, bf1) = val;
                }
            }
        }
    }
    
    return V;
}

Eigen::MatrixXd IntegralEngine::compute_core_hamiltonian() {
    /**
     * Compute core Hamiltonian: H = T + V
     * 
     * REFERENCE:
     * Szabo & Ostlund (1996), Eq. (3.152), p. 179
     */
    
    auto T = compute_kinetic();
    auto V = compute_nuclear();
    
    return T + V;
}

Eigen::Tensor<double, 4> IntegralEngine::compute_eri() {
    /**
     * Compute electron repulsion integrals using Libint2
     * 
     * Uses libint2::Engine with Operator::coulomb
     * 
     * ERI in chemist's notation: (μν|λσ)
     * 
     * ALGORITHM:
     * Loop over all unique shell quartets and compute ERI.
     * Uses 8-fold symmetry to reduce computation.
     */
    
    // Allocate ERI tensor using std::vector backing to avoid stack overflow
    size_t total_size = nbasis_ * nbasis_ * nbasis_ * nbasis_;
    std::vector<double> eri_data(total_size, 0.0);
    
    // Create TensorMap to wrap the data
    Eigen::TensorMap<Eigen::Tensor<double, 4>> ERI(
        eri_data.data(),
        nbasis_, nbasis_, nbasis_, nbasis_
    );
    
    // Find max number of primitives per shell
    size_t max_nprim = 0;
    for (const auto& shell : libint_shells_->shells) {
        max_nprim = std::max(max_nprim, shell.nprim());
    }
    
    libint2::Engine engine(libint2::Operator::coulomb,
                          max_nprim,
                          basis_.max_angular_momentum());
    
    const auto& shells = libint_shells_->shells;
    auto shell2bf = basis_.shell_to_basis_function_map();
    
    size_t nshells = shells.size();
    
    // Loop over shell quartets with 8-fold symmetry
    for (size_t s1 = 0; s1 < nshells; s1++) {
        size_t bf1_first = shell2bf[s1];
        size_t n1 = shells[s1].size();
        
        for (size_t s2 = 0; s2 <= s1; s2++) {
            size_t bf2_first = shell2bf[s2];
            size_t n2 = shells[s2].size();
            
            for (size_t s3 = 0; s3 <= s1; s3++) {
                size_t bf3_first = shell2bf[s3];
                size_t n3 = shells[s3].size();
                
                size_t s4_max = (s1 == s3) ? s2 : s3;
                for (size_t s4 = 0; s4 <= s4_max; s4++) {
                    size_t bf4_first = shell2bf[s4];
                    size_t n4 = shells[s4].size();
                    
                    // Compute shell quartet
                    engine.compute(shells[s1], shells[s2], 
                                 shells[s3], shells[s4]);
                    const auto& buf = engine.results();
                    
                    if (buf[0] == nullptr)
                        continue;
                    
                    // Extract to tensor with full symmetry
                    for (size_t f1 = 0; f1 < n1; f1++) {
                        size_t bf1 = bf1_first + f1;
                        
                        for (size_t f2 = 0; f2 < n2; f2++) {
                            size_t bf2 = bf2_first + f2;
                            
                            for (size_t f3 = 0; f3 < n3; f3++) {
                                size_t bf3 = bf3_first + f3;
                                
                                for (size_t f4 = 0; f4 < n4; f4++) {
                                    size_t bf4 = bf4_first + f4;
                                    
                                    size_t idx = f1 * n2 * n3 * n4 + 
                                               f2 * n3 * n4 + 
                                               f3 * n4 + f4;
                                    double val = buf[0][idx];
                                    
                                    // Apply 8-fold symmetry
                                    ERI(bf1, bf2, bf3, bf4) = val;
                                    ERI(bf2, bf1, bf3, bf4) = val;
                                    ERI(bf1, bf2, bf4, bf3) = val;
                                    ERI(bf2, bf1, bf4, bf3) = val;
                                    ERI(bf3, bf4, bf1, bf2) = val;
                                    ERI(bf4, bf3, bf1, bf2) = val;
                                    ERI(bf3, bf4, bf2, bf1) = val;
                                    ERI(bf4, bf3, bf2, bf1) = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Copy TensorMap data to owning Tensor before returning
    Eigen::Tensor<double, 4> result(nbasis_, nbasis_, nbasis_, nbasis_);
    for (size_t i = 0; i < total_size; i++) {
        result.data()[i] = eri_data[i];
    }
    
    return result;
}

Eigen::Tensor<double, 4> IntegralEngine::compute_eri_screened(double threshold) {
    /**
     * Compute ERI with Schwarz screening
     * 
     * Schwarz inequality: |(μν|λσ)| ≤ √[(μν|μν)(λσ|λσ)]
     * 
     * REFERENCE:
     * Häser & Ahlrichs, J. Comput. Chem. 10, 104 (1989)
     */
    
    // For simplicity, delegate to unscreened version
    // TODO: Implement proper Schwarz screening for large basis sets
    return compute_eri();
}

void IntegralEngine::print_statistics() const {
    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "        INTEGRAL ENGINE STATISTICS\n";
    std::cout << "============================================\n";
    std::cout << "\nBasis functions: " << nbasis_ << "\n";
    std::cout << "Number of shells: " << basis_.n_shells() << "\n";
    std::cout << "Max angular momentum: " << basis_.max_angular_momentum() << "\n";
    std::cout << "\nIntegral counts:\n";
    std::cout << "  One-electron (S, T, V): " << nbasis_ * nbasis_ << "\n";
    std::cout << "  Two-electron (ERI):     " << nbasis_ * nbasis_ * nbasis_ * nbasis_ << "\n";
    std::cout << "\nLibint2 version: " << LIBINT_VERSION << "\n";
    std::cout << "============================================\n\n";
}

// ============================================================================
// Free Functions
// ============================================================================

Eigen::MatrixXd compute_fock_matrix(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& P,
    const Eigen::Tensor<double, 4>& ERI) {
    /**
     * Compute Fock matrix for closed-shell systems
     * 
     * F_μν = H_μν + G_μν
     *      = H_μν + Σ_λσ P_λσ [2(μν|λσ) - (μλ|νσ)]
     * 
     * DERIVATION:
     * The two-electron contribution G comes from:
     * - Coulomb operator J: J_μν = Σ_λσ P_λσ (μν|λσ)
     * - Exchange operator K: K_μν = Σ_λσ P_λσ (μλ|νσ)
     * - G = 2J - K (factor of 2 for closed-shell)
     * 
     * REFERENCE:
     * Szabo & Ostlund (1996), Eq. (3.154), p. 139
     */
    
    size_t nbasis = H.rows();
    Eigen::MatrixXd F = H;  // Start with core Hamiltonian
    
    // Add two-electron contribution
    for (size_t mu = 0; mu < nbasis; mu++) {
        for (size_t nu = 0; nu < nbasis; nu++) {
            double G_mu_nu = 0.0;
            
            for (size_t lambda = 0; lambda < nbasis; lambda++) {
                for (size_t sigma = 0; sigma < nbasis; sigma++) {
                    // Coulomb: 2 * P_λσ * (μν|λσ)
                    G_mu_nu += 2.0 * P(lambda, sigma) * 
                              ERI(mu, nu, lambda, sigma);
                    
                    // Exchange: -P_λσ * (μλ|νσ)
                    G_mu_nu -= P(lambda, sigma) * 
                              ERI(mu, lambda, nu, sigma);
                }
            }
            
            F(mu, nu) += G_mu_nu;
        }
    }
    
    return F;
}

Eigen::Tensor<double, 4> transform_eri_to_mo(
    const Eigen::Tensor<double, 4>& ERI_AO,
    const Eigen::MatrixXd& C) {
    /**
     * Transform ERIs from AO to MO basis
     * 
     * (pq|rs)_MO = Σ_μνλσ C_μp C_νq C_λr C_σs (μν|λσ)_AO
     * 
     * ALGORITHM: Quarter transformations (4 steps)
     * This is more efficient than direct transformation (O(N^5) vs O(N^8))
     * 
     * REFERENCE:
     * Helgaker et al. (2000), Section 10.7, pp. 409-413
     */
    
    size_t nbasis = C.rows();
    size_t nmo = C.cols();
    
    Eigen::Tensor<double, 4> ERI_MO(nmo, nmo, nmo, nmo);
    ERI_MO.setZero();
    
    // Temporary tensors for intermediate steps
    Eigen::Tensor<double, 4> Temp1(nmo, nbasis, nbasis, nbasis);
    Eigen::Tensor<double, 4> Temp2(nmo, nmo, nbasis, nbasis);
    Eigen::Tensor<double, 4> Temp3(nmo, nmo, nmo, nbasis);
    
    Temp1.setZero();
    Temp2.setZero();
    Temp3.setZero();
    
    // Step 1: Transform first index
    for (size_t p = 0; p < nmo; p++) {
        for (size_t nu = 0; nu < nbasis; nu++) {
            for (size_t lambda = 0; lambda < nbasis; lambda++) {
                for (size_t sigma = 0; sigma < nbasis; sigma++) {
                    for (size_t mu = 0; mu < nbasis; mu++) {
                        Temp1(p, nu, lambda, sigma) += 
                            C(mu, p) * ERI_AO(mu, nu, lambda, sigma);
                    }
                }
            }
        }
    }
    
    // Step 2: Transform second index
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t lambda = 0; lambda < nbasis; lambda++) {
                for (size_t sigma = 0; sigma < nbasis; sigma++) {
                    for (size_t nu = 0; nu < nbasis; nu++) {
                        Temp2(p, q, lambda, sigma) += 
                            C(nu, q) * Temp1(p, nu, lambda, sigma);
                    }
                }
            }
        }
    }
    
    // Step 3: Transform third index
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t sigma = 0; sigma < nbasis; sigma++) {
                    for (size_t lambda = 0; lambda < nbasis; lambda++) {
                        Temp3(p, q, r, sigma) += 
                            C(lambda, r) * Temp2(p, q, lambda, sigma);
                    }
                }
            }
        }
    }
    
    // Step 4: Transform fourth index
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = 0; s < nmo; s++) {
                    for (size_t sigma = 0; sigma < nbasis; sigma++) {
                        ERI_MO(p, q, r, s) += 
                            C(sigma, s) * Temp3(p, q, r, sigma);
                    }
                }
            }
        }
    }
    
    return ERI_MO;
}

// ============================================================================
// DF-MP2 Integrals (3-center and 2-center)
// ============================================================================

Eigen::Tensor<double, 3> IntegralEngine::compute_3center_eri(
    const BasisSet& aux_basis
) {
    /**
     * Compute 3-center electron repulsion integrals for DF-MP2
     * 
     * Formula: (μν|P) = ∫∫ φ_μ(r1)φ_ν(r1) r12^-1 χ_P(r2) dr1 dr2
     * 
     * where μ,ν are primary basis functions and P is auxiliary basis.
     * These are used in density-fitting approximation:
     *   (μν|λσ) ≈ Σ_PQ (μν|P) [J^-1]_PQ (Q|λσ)
     * 
     * REFERENCE:
     * F. Weigend & M. Häser, Theor. Chem. Acc. 97, 331 (1997), Eq. (2)
     * "RI-MP2: first derivatives and global consistency"
     * 
     * @param aux_basis Auxiliary basis set (e.g. cc-pVTZ-RI)
     * @return 3-center tensor [nbasis × nbasis × naux]
     */
    
    size_t naux = aux_basis.n_basis_functions();
    
    std::cout << "Computing 3-center integrals (μν|P)...\n";
    std::cout << "  Primary basis: " << nbasis_ << " functions\n";
    std::cout << "  Auxiliary basis: " << naux << " functions\n";
    
    // Using Libint2's BraKet::xs_xx mode for TRUE 3-center integrals!
    // Technique from libint2/lcao/1body.h (libint-master)
    // REFERENCE: Valeev (2014), Libint2 documentation
    std::cout << "  Using Libint2 BraKet::xs_xx for true 3-center integrals\n";
    
    // Allocate 3-center tensor
    std::vector<double> data(nbasis_ * nbasis_ * naux, 0.0);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> B(
        data.data(), nbasis_, nbasis_, naux
    );
    
    // Convert auxiliary basis to libint2 format
    std::vector<libint2::Shell> aux_shells;
    aux_shells.reserve(aux_basis.n_shells());
    
    for (size_t i = 0; i < aux_basis.n_shells(); i++) {
        const auto& shell = aux_basis.shell(i);
        int l = shell.l();
        auto pos = shell.position();
        std::array<double, 3> center = {pos[0], pos[1], pos[2]};
        
        std::vector<double> exps, coeffs;
        for (size_t j = 0; j < shell.n_primitives(); j++) {
            const auto& prim = shell.primitive(j);
            exps.push_back(prim.exponent);
            coeffs.push_back(prim.coefficient);
        }
        
        libint2::svector<double> alpha_sv(exps.begin(), exps.end());
        libint2::svector<double> coeff_sv(coeffs.begin(), coeffs.end());
        
        libint2::Shell::Contraction contr;
        contr.l = l;
        contr.pure = shell.is_spherical();
        contr.coeff = coeff_sv;
        
        libint2::Shell libint_shell(alpha_sv, {{contr}}, center);
        aux_shells.push_back(std::move(libint_shell));
    }
    
    // Find max parameters
    size_t max_nprim_aux = 0;
    int max_l_aux = 0;
    for (const auto& shell : aux_shells) {
        max_nprim_aux = std::max(max_nprim_aux, shell.nprim());
        max_l_aux = std::max(max_l_aux, shell.contr[0].l);
    }
    
    // Don't oversize - libint2 preallocates memory
    size_t max_nprim_prim = 0;
    int max_l_prim = 0;
    for (const auto& shell : libint_shells_->shells) {
        max_nprim_prim = std::max(max_nprim_prim, shell.nprim());
        max_l_prim = std::max(max_l_prim, shell.contr[0].l);
    }
    
    size_t max_nprim = std::max(max_nprim_prim, max_nprim_aux);
    int max_l = std::max(max_l_prim, max_l_aux);
    
    // TRUE 3-CENTER INTEGRALS using Libint2's BraKet::xs_xx mode
    // System libint2 v2.9.0 has INCLUDE_ERI3=2 support!
    //
    // ALGORITHM:
    // 1. Create engine in default mode (xx_xx)
    // 2. Set to xs_xx mode: (aux|unit) (primary|primary)
    // 3. Compute (P|μν) directly - no approximation!
    //
    // REFERENCE:
    // - Valeev, E. F. Libint2 (2024), engine.impl.h line 1826-1981
    // - Weigend & Häser (1997), Eq. (2): definition of 3-center integrals
    
    // Create libint2 engine and set to xs_xx (3-center) mode
    libint2::Engine engine(
        libint2::Operator::coulomb,
        max_nprim,
        max_l,
        0  // no derivatives
    );
    
    // Set to 3-center mode
    engine.set(libint2::BraKet::xs_xx);
    std::cout << "  Engine mode: xs_xx (TRUE 3-center integrals)\n";
    
    const auto& prim_shells = libint_shells_->shells;
    auto shell2bf_prim = basis_.shell_to_basis_function_map();
    auto shell2bf_aux = aux_basis.shell_to_basis_function_map();
    
    // Loop over shell triplets: auxiliary P, primary μ, primary ν
    // xs_xx mode: engine.compute(aux_shell, prim_shell_1, prim_shell_2)
    // Result: (P|μν) with P from aux, μν from primary
    size_t n_computed = 0;
    for (size_t sP = 0; sP < aux_shells.size(); sP++) {
        size_t bfP_first = shell2bf_aux[sP];
        size_t nP = aux_shells[sP].size();
        
        for (size_t s1 = 0; s1 < prim_shells.size(); s1++) {
            size_t bf1_first = shell2bf_prim[s1];
            size_t n1 = prim_shells[s1].size();
            
            for (size_t s2 = 0; s2 <= s1; s2++) {
                size_t bf2_first = shell2bf_prim[s2];
                size_t n2 = prim_shells[s2].size();
                
                // Compute TRUE 3-center (P|μν) with xs_xx mode
                // xs_xx convention: single shell first, then pair
                engine.compute(aux_shells[sP], prim_shells[s1], prim_shells[s2]);
                
                const auto& buf = engine.results();
                if (buf[0] == nullptr) continue;
                
                n_computed++;
                
                // Extract (P|μν) - no averaging needed, this is exact!
                // Layout: buf[0][fP * n1*n2 + f1*n2 + f2]
                for (size_t fP = 0; fP < nP; fP++) {
                    for (size_t f1 = 0; f1 < n1; f1++) {
                        for (size_t f2 = 0; f2 < n2; f2++) {
                            size_t bf1 = bf1_first + f1;
                            size_t bf2 = bf2_first + f2;
                            size_t bfP = bfP_first + fP;
                            
                            // Index for xs_xx: P is leading dimension
                            size_t idx = fP * (n1 * n2) + f1 * n2 + f2;
                            double val = buf[0][idx];
                            
                            // Store with μν symmetry
                            B(bf1, bf2, bfP) = val;
                            if (bf1 != bf2) {
                                B(bf2, bf1, bfP) = val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "  Computed " << n_computed << " shell triplets\n";
    std::cout << "  3-center integrals ready (weighted average - improved)\n";
    
    // Copy to owning tensor
    Eigen::Tensor<double, 3> result(nbasis_, nbasis_, naux);
    for (size_t i = 0; i < nbasis_ * nbasis_ * naux; i++) {
        result.data()[i] = data[i];
    }
    
    return result;
}

Eigen::MatrixXd IntegralEngine::compute_2center_eri(
    const BasisSet& aux_basis
) {
    /**
     * Compute 2-center auxiliary metric for DF-MP2
     * 
     * Formula: J_PQ = (P|Q) = ∫∫ χ_P(r1) r12^-1 χ_Q(r2) dr1 dr2
     * 
     * This metric is inverted to form J^-1 in density-fitting:
     *   (μν|λσ) ≈ Σ_PQ (μν|P) [J^-1]_PQ (Q|λσ)
     * 
     * The metric must be positive definite (all eigenvalues > 0)
     * for stable inversion via Cholesky decomposition.
     * 
     * REFERENCE:
     * M. Feyereisen et al., Chem. Phys. Lett. 208, 359 (1993), Eq. (8)
     * "Use of approximate integrals in ab initio theory"
     * 
     * @param aux_basis Auxiliary basis set
     * @return Metric matrix [naux × naux]
     */
    
    size_t naux = aux_basis.n_basis_functions();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(naux, naux);
    
    std::cout << "Computing 2-center metric (P|Q)...\n";
    std::cout << "  Auxiliary basis: " << naux << " functions\n";
    
    // Convert auxiliary basis
    std::vector<libint2::Shell> aux_shells;
    aux_shells.reserve(aux_basis.n_shells());
    
    for (size_t i = 0; i < aux_basis.n_shells(); i++) {
        const auto& shell = aux_basis.shell(i);
        int l = shell.l();
        auto pos = shell.position();
        std::array<double, 3> center = {pos[0], pos[1], pos[2]};
        
        std::vector<double> exps, coeffs;
        for (size_t j = 0; j < shell.n_primitives(); j++) {
            const auto& prim = shell.primitive(j);
            exps.push_back(prim.exponent);
            coeffs.push_back(prim.coefficient);
        }
        
        libint2::svector<double> alpha_sv(exps.begin(), exps.end());
        libint2::svector<double> coeff_sv(coeffs.begin(), coeffs.end());
        
        libint2::Shell::Contraction contr;
        contr.l = l;
        contr.pure = shell.is_spherical();
        contr.coeff = coeff_sv;
        
        libint2::Shell libint_shell(alpha_sv, {{contr}}, center);
        aux_shells.push_back(std::move(libint_shell));
    }
    
    // Max parameters for aux only
    size_t max_nprim = 0;
    int max_l = 0;
    for (const auto& shell : aux_shells) {
        max_nprim = std::max(max_nprim, shell.nprim());
        max_l = std::max(max_l, shell.contr[0].l);
    }
    
    // Create engine (standard 4-center but all aux)
    libint2::Engine engine(
        libint2::Operator::coulomb,
        max_nprim,  // DON'T add 100!
        max_l
    );
    
    auto shell2bf = aux_basis.shell_to_basis_function_map();
    
    // Loop over aux shell pairs (P, Q)
    for (size_t sP = 0; sP < aux_shells.size(); sP++) {
        size_t bfP_first = shell2bf[sP];
        size_t nP = aux_shells[sP].size();
        
        for (size_t sQ = 0; sQ <= sP; sQ++) {
            size_t bfQ_first = shell2bf[sQ];
            size_t nQ = aux_shells[sQ].size();
            
            // Compute (sP sQ | sP sQ) - self-overlap
            engine.compute(aux_shells[sP], aux_shells[sQ],
                          aux_shells[sP], aux_shells[sQ]);
            
            const auto& buf = engine.results();
            if (buf[0] == nullptr) continue;
            
            // Extract diagonal blocks
            for (size_t fP = 0; fP < nP; fP++) {
                for (size_t fQ = 0; fQ < nQ; fQ++) {
                    size_t bfP = bfP_first + fP;
                    size_t bfQ = bfQ_first + fQ;
                    
                    // Index: (fP fQ | fP fQ)
                    size_t idx = fP * nQ * nP * nQ + 
                                fQ * nP * nQ + 
                                fP * nQ + fQ;
                    double val = buf[0][idx];
                    
                    J(bfP, bfQ) = val;
                    if (bfP != bfQ) {
                        J(bfQ, bfP) = val;  // symmetry
                    }
                }
            }
        }
    }
    
    std::cout << "  Metric computed\n";
    
    return J;
}

} // namespace mshqc
