// Author: Muhamad Syahrul Hidayat
// Date: 2025-11-16
//
// Transition density matrices for excited state properties in CI calculations
//
// Theory References:
// - Hirata, S. (2004). "Tensor Contraction Engine: Abstraction and Automated 
//   Parallel Implementation of Configuration-Interaction, Coupled-Cluster, 
//   and Many-Body Perturbation Theories"
//   J. Phys. Chem. A 107, 9887-9897.
//   DOI: 10.1021/jp034596w
//
// - Head-Gordon, M., Rico, R. J., Oumi, M., & Lee, T. J. (1995). 
//   "A doubles correction to electronic excited states from configuration interaction 
//   in the space of single substitutions"
//   Chem. Phys. Lett. 219, 21-29.
//   DOI: 10.1016/0009-2614(94)00070-0
//
// - Lischka, H., Dallos, M., & Shepard, R. (2002). 
//   "Analytic MRCI gradient for excited states: formalism and application 
//   to the n-π* valence- and n-(3s,3p) Rydberg states of formaldehyde"
//   Mol. Phys. 100, 1647-1658.
//   DOI: 10.1080/00268970210155121
//
// Transition Density Matrix (1-TDM):
//   γ_pq^IJ = <Ψ_I|a†_p a_q|Ψ_J>
//
// where:
//   Ψ_I, Ψ_J = CI wavefunctions for states I and J
//   a†_p, a_q = creation/annihilation operators for orbitals p, q
//
// For I=J: Regular 1-RDM (one-particle density matrix)
// For I≠J: Transition density between states I and J
//
// Applications:
//   - Transition dipole moments: μ_IJ = Tr(γ^IJ · μ)
//   - Oscillator strengths: f_IJ = (2/3) * ΔE_IJ * |μ_IJ|²
//   - Natural transition orbitals (NTOs)
//   - Excited state gradients
//
// ============================================================================
// ORIGINAL IMPLEMENTATION - NO CODE COPIED FROM PYSCF/PSI4
// ============================================================================

#ifndef MSHQC_CI_TRANSITION_DENSITY_H
#define MSHQC_CI_TRANSITION_DENSITY_H

#include <Eigen/Dense>
#include <vector>

namespace mshqc {
namespace ci {

/**
 * Transition density matrix for CI excited states
 * 
 * Computes 1-TDM between two CI states:
 *   γ_pq^IJ = <Ψ_I|a†_p a_q|Ψ_J>
 * 
 * Used for:
 *   - Transition dipole moments
 *   - Oscillator strengths
 *   - Natural transition orbitals (NTOs)
 *   - Excited state properties
 */
class TransitionDensity {
public:
    /**
     * Compute 1-TDM between two CI states
     * 
     * Uses Slater-Condon rules to evaluate:
     *   γ_pq = Σ_K Σ_L c_K^I c_L^J <K|a†_p a_q|L>
     * 
     * where K, L are determinants in the CI expansion
     * 
     * @param ci_coeff_i CI coefficients for state I
     * @param ci_coeff_j CI coefficients for state J
     * @param determinants List of determinants in CI space
     * @param n_orbitals Number of spatial orbitals
     * @return 1-TDM matrix (n_orbitals x n_orbitals)
     */
    static Eigen::MatrixXd compute_1tdm(
        const std::vector<double>& ci_coeff_i,
        const std::vector<double>& ci_coeff_j,
        const std::vector<class Determinant>& determinants,
        int n_orbitals
    );

    /**
     * Compute transition dipole moment between two states
     * 
     * μ_IJ = Tr(γ^IJ · μ)
     * 
     * where μ is the dipole integral matrix in AO or MO basis
     * 
     * @param tdm Transition density matrix γ^IJ
     * @param dipole_integrals Dipole integral matrix [x, y, z components]
     * @return Transition dipole vector [μ_x, μ_y, μ_z] in atomic units
     */
    static Eigen::Vector3d transition_dipole_moment(
        const Eigen::MatrixXd& tdm,
        const std::vector<Eigen::MatrixXd>& dipole_integrals
    );

    /**
     * Compute oscillator strength for electronic transition I → J
     * 
     * f_IJ = (2/3) * ΔE_IJ * |μ_IJ|²
     * 
     * where:
     *   ΔE_IJ = energy difference (Hartree)
     *   μ_IJ = transition dipole moment (a.u.)
     * 
     * Oscillator strength is dimensionless and measures transition intensity.
     * Sum rule: Σ_J f_IJ = N (number of electrons)
     * 
     * @param transition_dipole Transition dipole vector μ_IJ (a.u.)
     * @param delta_e Energy difference E_J - E_I (Hartree)
     * @return Oscillator strength f_IJ (dimensionless)
     */
    static double oscillator_strength(
        const Eigen::Vector3d& transition_dipole,
        double delta_e
    );

    /**
     * Natural transition orbitals (NTOs) via SVD
     * 
     * Decomposes transition density as:
     *   γ^IJ = U Σ V†
     * 
     * where:
     *   U = hole orbitals (where electron comes from)
     *   V = particle orbitals (where electron goes to)
     *   Σ = singular values (transition amplitudes)
     * 
     * NTOs provide compact representation of excitation character.
     * Typically 1-2 pairs dominate for simple excitations.
     * 
     * Reference: Martin, R. L. (2003). J. Chem. Phys. 118, 4775.
     * 
     * @param tdm Transition density matrix
     * @param hole_orbitals Output: hole NTOs (columns of U)
     * @param particle_orbitals Output: particle NTOs (columns of V)
     * @param amplitudes Output: singular values (transition amplitudes)
     */
    static void natural_transition_orbitals(
        const Eigen::MatrixXd& tdm,
        Eigen::MatrixXd& hole_orbitals,
        Eigen::MatrixXd& particle_orbitals,
        Eigen::VectorXd& amplitudes
    );

    /**
     * Attachment/detachment densities for excitation analysis
     * 
     * Attachment density (where electron goes):
     *   ρ_attach = diagonal of γ^IJ · (γ^IJ)†
     * 
     * Detachment density (where electron comes from):
     *   ρ_detach = diagonal of (γ^IJ)† · γ^IJ
     * 
     * Used for visualizing excitation character in real space.
     * 
     * @param tdm Transition density matrix
     * @param attachment Output: attachment density (n_orbitals vector)
     * @param detachment Output: detachment density (n_orbitals vector)
     */
    static void attachment_detachment_density(
        const Eigen::MatrixXd& tdm,
        Eigen::VectorXd& attachment,
        Eigen::VectorXd& detachment
    );

    /**
     * Check if transition is electric-dipole allowed
     * 
     * Selection rules for electric dipole transitions:
     *   - ΔS = 0 (no spin change, singlet-singlet or triplet-triplet)
     *   - |μ_IJ| > threshold (non-zero transition dipole)
     * 
     * @param transition_dipole Transition dipole moment
     * @param threshold Minimum dipole magnitude (default 1e-6 a.u.)
     * @return True if transition is dipole-allowed
     */
    static bool is_dipole_allowed(
        const Eigen::Vector3d& transition_dipole,
        double threshold = 1e-6
    );

    /**
     * Compute Einstein A coefficient for spontaneous emission
     * 
     * A_IJ = (2 * ω³_IJ * |μ_IJ|²) / (3 * c³)
     * 
     * where:
     *   ω_IJ = transition frequency (a.u.)
     *   μ_IJ = transition dipole (a.u.)
     *   c = speed of light (a.u.)
     * 
     * Returns rate in atomic units (inverse time).
     * Convert to s⁻¹ by multiplying by 4.134×10¹⁶.
     * 
     * @param transition_dipole Transition dipole moment (a.u.)
     * @param delta_e Energy difference (Hartree)
     * @return Einstein A coefficient (a.u. time⁻¹)
     */
    static double einstein_a_coefficient(
        const Eigen::Vector3d& transition_dipole,
        double delta_e
    );

private:
    // Speed of light in atomic units
    static constexpr double SPEED_OF_LIGHT_AU = 137.035999084;
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_TRANSITION_DENSITY_H
