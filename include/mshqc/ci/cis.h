/**
 * @file cis.h
 * @brief CIS (Configuration Interaction Singles) for excited states
 * 
 * Implements singles-only CI for electronic excitation energies.
 * Used for UV-Vis spectra calculations.
 * 
 * THEORY REFERENCES:
 *   - Foresman et al. (1992), J. Phys. Chem. 96, 135
 *   - Szabo & Ostlund (1996), Ch. 4
 *   - Helgaker et al. (2000), Ch. 10
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 * @license MIT
 * 
 * @note Original implementation from published CASSCF theory.
 *       No code copied from existing quantum chemistry software.
 */

#ifndef MSHQC_CI_CIS_H
#define MSHQC_CI_CIS_H

#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include "mshqc/ci/davidson.h"
#include <vector>

namespace mshqc {
namespace ci {

/**
 * CIS excitation result
 */
struct CISExcitation {
    double energy;               // Excitation energy (Ha)
    double wavelength;           // Wavelength (nm)
    double oscillator_strength;  // Oscillator strength
    Eigen::VectorXd amplitudes;  // CI coefficients
    
    // Dominant transitions
    struct Transition {
        int i;         // Occupied orbital
        int a;         // Virtual orbital
        bool alpha;    // Spin
        double coeff;  // Amplitude
    };
    std::vector<Transition> dominant;
};

/**
 * CIS result structure
 */
struct CISResult {
    double ground_state;              // HF ground state energy
    std::vector<CISExcitation> excitations;  // Excited states
    int n_states;                     // # of excited states
    bool converged;                   // Convergence flag
};

/**
 * CIS (Configuration Interaction Singles)
 * 
 * THEORY: Only single excitations from HF reference
 * |Ψ⟩ = c_0 |HF⟩ + Σ_ia c_i^a |Φ_i^a⟩
 * 
 * Good for:
 * - Valence excitations
 * - UV-Vis spectra
 * - Vertical excitation energies
 * 
 * Limitations:
 * - No correlation (similar to HF for ground state)
 * - Tends to overestimate excitation energies
 * - Missing double excitations
 */
class CIS {
public:
    /**
     * Constructor
     * @param hf_result HF reference wavefunction
     * @param ints MO integrals
     */
    CIS(const CIIntegrals& ints,
        const Determinant& hf_det,
        int n_occ_alpha, int n_occ_beta,
        int n_virt_alpha, int n_virt_beta);
    
    /**
     * Compute CIS excitation energies
     * 
     * REFERENCE: Foresman et al. (1992), J. Phys. Chem. 96, 135
     * 
     * @param n_states Number of excited states to compute
     * @return CIS result with excitation energies
     */
    CISResult compute(int n_states = 5);
    
    /**
     * Get CIS determinants (HF + all singles)
     */
    std::vector<Determinant> get_determinants() const;
    
private:
    const CIIntegrals& ints_;
    Determinant hf_det_;
    int nocc_a_, nocc_b_;
    int nvirt_a_, nvirt_b_;
    
    /**
     * Generate all single excitations from HF
     * REFERENCE: Szabo & Ostlund (1996), Section 4.2
     */
    std::vector<Determinant> generate_singles();
    
    /**
     * Compute oscillator strength
     * REFERENCE: Helgaker et al. (2000), Eq. (10.148)
     * f = (2/3) * ΔE * |⟨Ψ_0|μ|Ψ_i⟩|²
     */
    double compute_oscillator_strength(
        const Eigen::VectorXd& ground,
        const Eigen::VectorXd& excited,
        double delta_e);
    
    /**
     * Extract dominant transitions from CI vector
     */
    std::vector<CISExcitation::Transition> get_dominant_transitions(
        const std::vector<Determinant>& dets,
        const Eigen::VectorXd& c,
        double threshold = 0.1);
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_CIS_H
