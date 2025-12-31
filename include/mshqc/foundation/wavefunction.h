/**
 * @file wavefunction.h
 * @brief Universal wavefunction container for correlated methods
 * 
 * Storage and analysis of wavefunction amplitudes from MP, CI, CC, and MCSCF methods.
 * 
 * THEORY:
 * Correlated wavefunction expansion:
 *   |Ψ⟩ = |Ψ_HF⟩ + |Ψ^(1)⟩ + |Ψ^(2)⟩ + |Ψ^(3)⟩ + ...
 * 
 * where:
 *   |Ψ^(k)⟩ = Σ_ijab t_ij^ab(k) |Φ_ij^ab⟩  (k-th order correction)
 * 
 * REFERENCES:
 *   - C. Møller & M. S. Plesset, Phys. Rev. 46, 618 (1934) [MP theory]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996), Ch. 6
 *   - T. Helgaker et al., "Molecular Electronic-Structure Theory" (2000), Ch. 10
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 */

#ifndef MSHQC_FOUNDATION_WAVEFUNCTION_H
#define MSHQC_FOUNDATION_WAVEFUNCTION_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#include <iostream>

namespace mshqc {
namespace foundation {

/**
 * @brief Single excitation amplitude with orbital indices
 * 
 * Represents: t_ij^ab |Φ_ij^ab⟩
 * where i,j = occupied, a,b = virtual, spin = αα, ββ, or αβ
 */
struct Excitation {
    int i, j;           // Occupied orbital indices
    int a, b;           // Virtual orbital indices
    double amplitude;   // t_ij^ab coefficient
    std::string spin;   // "aa", "bb", or "ab"
    
    // Sort by absolute amplitude (largest first)
    bool operator<(const Excitation& other) const {
        return std::abs(amplitude) > std::abs(other.amplitude);
    }
    
    // For printing
    std::string to_string() const;
};

/**
 * @brief Universal wavefunction container for correlated methods
 * 
 * Stores T2 amplitudes from MP, CI, CC methods:
 *   - MP2: T2^(1) amplitudes
 *   - MP3: T2^(2) corrections
 *   - MP4: T2^(3) corrections
 *   - CI/CC: configuration coefficients
 * 
 * Supports unrestricted (UHF) reference:
 *   - T2_aa: α-α spin case
 *   - T2_bb: β-β spin case
 *   - T2_ab: α-β spin case
 * 
 * For restricted (RHF) reference: only need T2_aa (β = α)
 */
class Wavefunction {
public:
    /**
     * @brief Construct wavefunction container
     * @param nocc_a Number of occupied α orbitals
     * @param nocc_b Number of occupied β orbitals
     * @param nvirt_a Number of virtual α orbitals
     * @param nvirt_b Number of virtual β orbitals
     */
    Wavefunction(int nocc_a, int nocc_b, int nvirt_a, int nvirt_b);
    
    /**
     * @brief Set first-order T2 amplitudes (from MP2, CCSD)
     * @param t2_aa α-α amplitudes (nocc_a × nocc_a × nvirt_a × nvirt_a)
     * @param t2_bb β-β amplitudes (nocc_b × nocc_b × nvirt_b × nvirt_b)
     * @param t2_ab α-β amplitudes (nocc_a × nocc_b × nvirt_a × nvirt_b)
     */
    void set_t2_order_1(const Eigen::Tensor<double, 4>& t2_aa,
                        const Eigen::Tensor<double, 4>& t2_bb,
                        const Eigen::Tensor<double, 4>& t2_ab);
    
    /**
     * @brief Set second-order T2 corrections (from MP3)
     */
    void set_t2_order_2(const Eigen::Tensor<double, 4>& t2_aa,
                        const Eigen::Tensor<double, 4>& t2_bb,
                        const Eigen::Tensor<double, 4>& t2_ab);
    
    /**
     * @brief Set third-order T2 corrections (from MP4)
     */
    void set_t2_order_3(const Eigen::Tensor<double, 4>& t2_aa,
                        const Eigen::Tensor<double, 4>& t2_bb,
                        const Eigen::Tensor<double, 4>& t2_ab);
    
    /**
     * @brief Get T2 amplitudes by order and spin
     * @param order Perturbation order (1=MP2, 2=MP3, 3=MP4)
     * @param spin Spin case ("aa", "bb", "ab")
     * @return Reference to T2 tensor
     */
    const Eigen::Tensor<double, 4>& get_t2(int order, const std::string& spin) const;
    
    /**
     * @brief Extract dominant excitations above threshold
     * @param threshold Minimum |t_ij^ab| to include (default: 0.05)
     * @param max_excitations Maximum number to return (default: 20)
     * @return Vector of excitations sorted by amplitude
     * 
     * USAGE: Find which orbital excitations contribute most to correlation
     * Example: HOMO→LUMO, HOMO-1→LUMO+1, etc.
     */
    std::vector<Excitation> dominant_amplitudes(double threshold = 0.05,
                                                 int max_excitations = 20) const;
    
    /**
     * @brief Calculate wavefunction norm squared ⟨Ψ|Ψ⟩
     * @param order Which order to include (1=MP2 only, 2=MP2+MP3, 3=all)
     * @return Norm squared
     * 
     * For normalized wavefunction: ⟨Ψ|Ψ⟩ = 1 + Σ |t_ij^ab|²
     */
    double norm_squared(int order = 1) const;
    
    /**
     * @brief Print summary of wavefunction
     * @param os Output stream
     */
    void print_summary(std::ostream& os = std::cout) const;
    
    /**
     * @brief Print wavefunction in LaTeX format for thesis/papers
     * @param os Output stream
     * @param max_terms Maximum number of terms to show
     * 
     * Output format:
     *   |Ψ⟩ = |Ψ_HF⟩ + 0.287 |Φ_{1,1}^{2,2}⟩(αα) - 0.156 |Φ_{1,1}^{2,3}⟩(αβ) + ...
     */
    void print_latex(std::ostream& os, int max_terms = 20) const;
    
    /**
     * @brief Check if wavefunction has amplitudes stored
     */
    bool has_amplitudes(int order = 1) const;
    
    /**
     * @brief Get orbital dimensions
     */
    int nocc_alpha() const { return nocc_a_; }
    int nocc_beta() const { return nocc_b_; }
    int nvirt_alpha() const { return nvirt_a_; }
    int nvirt_beta() const { return nvirt_b_; }
    
private:
    // Orbital dimensions
    int nocc_a_, nocc_b_;    // Occupied α, β
    int nvirt_a_, nvirt_b_;  // Virtual α, β
    
    // T2 amplitudes by order (0 = empty)
    // Order 1: MP2, Order 2: MP3, Order 3: MP4
    Eigen::Tensor<double, 4> t2_1_aa_, t2_1_bb_, t2_1_ab_;  // First-order
    Eigen::Tensor<double, 4> t2_2_aa_, t2_2_bb_, t2_2_ab_;  // Second-order
    Eigen::Tensor<double, 4> t2_3_aa_, t2_3_bb_, t2_3_ab_;  // Third-order
    
    /**
     * @brief Extract excitations from a single T2 tensor
     */
    std::vector<Excitation> extract_from_tensor(
        const Eigen::Tensor<double, 4>& t2,
        const std::string& spin,
        double threshold) const;
};

/**
 * @brief Calculate one-particle density matrix (OPDM) from T2 amplitudes
 * 
 * REFERENCE: Helgaker et al. (2000), Section 10.4
 * 
 * OPDM elements:
 *   γ_pq = ⟨Ψ| p^† q |Ψ⟩
 * 
 * From T2^(1) (MP2 level):
 *   γ_ij = δ_ij - 0.5 * Σ_kab t_ik^ab t_jk^ab  (occ-occ)
 *   γ_ab = 0.5 * Σ_ijc t_ij^ac t_ij^bc         (virt-virt)
 * 
 * @param wfn Wavefunction with T2 amplitudes
 * @return OPDM matrix (nbasis × nbasis)
 */
Eigen::MatrixXd compute_opdm(const Wavefunction& wfn);

/**
 * @brief Calculate natural orbitals from wavefunction
 * 
 * Natural orbitals are eigenvectors of OPDM:
 *   γ = U n U^†
 * 
 * where n = occupation numbers (0 to 2)
 * 
 * REFERENCE: P.-O. Löwdin, Phys. Rev. 97, 1474 (1955)
 * 
 * @param wfn Wavefunction
 * @return Pair of (natural orbitals, occupation numbers)
 */
std::pair<Eigen::MatrixXd, Eigen::VectorXd> 
compute_natural_orbitals(const Wavefunction& wfn);

} // namespace foundation
} // namespace mshqc

#endif // MSHQC_FOUNDATION_WAVEFUNCTION_H
