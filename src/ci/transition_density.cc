// Author: Muhamad Syahrul Hidayat
// Date: 2025-11-16
//
// Implementation of transition density matrices for CI excited states
//
// ============================================================================
// ORIGINAL IMPLEMENTATION - NO CODE COPIED FROM PYSCF/PSI4
// ============================================================================

#include "mshqc/ci/transition_density.h"
#include "mshqc/ci/determinant.h"
#include <cmath>
#include <stdexcept>
#include <Eigen/SVD>

namespace mshqc {
namespace ci {

Eigen::MatrixXd TransitionDensity::compute_1tdm(
    const std::vector<double>& ci_coeff_i,
    const std::vector<double>& ci_coeff_j,
    const std::vector<class Determinant>& determinants,
    int n_orbitals
) {
    size_t n_det = determinants.size();
    
    if (ci_coeff_i.size() != n_det || ci_coeff_j.size() != n_det) {
        throw std::invalid_argument(
            "TransitionDensity::compute_1tdm: "
            "CI coefficient size mismatch with determinant list"
        );
    }

    // Initialize 1-TDM: γ_pq^IJ = <Ψ_I|a†_p a_q|Ψ_J>
    Eigen::MatrixXd tdm = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);

    // Loop over all determinant pairs
    for (size_t k_idx = 0; k_idx < n_det; ++k_idx) {
        for (size_t l_idx = 0; l_idx < n_det; ++l_idx) {
            const Determinant& det_k = determinants[k_idx];
            const Determinant& det_l = determinants[l_idx];

            double c_k = ci_coeff_i[k_idx];
            double c_l = ci_coeff_j[l_idx];

            // Skip if coefficients are negligible
            if (std::abs(c_k * c_l) < 1e-12) continue;

            // Compute <K|a†_p a_q|L> using Slater-Condon rules
            // Only non-zero if K and L differ by at most one orbital
            auto [n_diff_alpha, n_diff_beta] = det_k.excitation_level(det_l);

            if (n_diff_alpha + n_diff_beta == 0) {
                // Same determinant: diagonal elements
                // <K|a†_p a_q|K> = δ_pq if orbital p is occupied in K
                std::vector<int> alpha_occ = det_k.alpha_occupations();
                std::vector<int> beta_occ = det_k.beta_occupations();

                for (int p : alpha_occ) {
                    tdm(p, p) += c_k * c_l;
                }
                for (int p : beta_occ) {
                    tdm(p, p) += c_k * c_l;
                }

            } else if (n_diff_alpha == 1 && n_diff_beta == 0) {
                // Single excitation in α spin
                // L = a†_a a_i K  →  <K|a†_i a_a|L> = phase
                std::vector<int> occ_k = det_k.alpha_occupations();
                std::vector<int> occ_l = det_l.alpha_occupations();

                // Find differing orbital
                int i = -1, a = -1;
                for (int orb : occ_k) {
                    if (!det_l.is_occupied(orb, true)) {
                        i = orb; break;
                    }
                }
                for (int orb : occ_l) {
                    if (!det_k.is_occupied(orb, true)) {
                        a = orb; break;
                    }
                }

                if (i >= 0 && a >= 0) {
                    int phase = det_l.phase(i, a, true);
                    tdm(i, a) += c_k * c_l * phase;
                }

            } else if (n_diff_alpha == 0 && n_diff_beta == 1) {
                // Single excitation in β spin
                std::vector<int> occ_k = det_k.beta_occupations();
                std::vector<int> occ_l = det_l.beta_occupations();

                int i = -1, a = -1;
                for (int orb : occ_k) {
                    if (!det_l.is_occupied(orb, false)) {
                        i = orb; break;
                    }
                }
                for (int orb : occ_l) {
                    if (!det_k.is_occupied(orb, false)) {
                        a = orb; break;
                    }
                }

                if (i >= 0 && a >= 0) {
                    int phase = det_l.phase(i, a, false);
                    tdm(i, a) += c_k * c_l * phase;
                }
            }
            // Higher excitations: matrix element = 0
        }
    }

    return tdm;
}

Eigen::Vector3d TransitionDensity::transition_dipole_moment(
    const Eigen::MatrixXd& tdm,
    const std::vector<Eigen::MatrixXd>& dipole_integrals
) {
    if (dipole_integrals.size() != 3) {
        throw std::invalid_argument(
            "TransitionDensity::transition_dipole_moment: "
            "Dipole integrals must have 3 components (x, y, z)"
        );
    }

    // Transition dipole: μ_IJ = Tr(γ^IJ · μ)
    Eigen::Vector3d mu_trans;
    
    for (int dir = 0; dir < 3; ++dir) {
        const Eigen::MatrixXd& mu = dipole_integrals[dir];
        
        if (mu.rows() != tdm.rows() || mu.cols() != tdm.cols()) {
            throw std::invalid_argument(
                "TransitionDensity::transition_dipole_moment: "
                "Dipole integral matrix size mismatch with TDM"
            );
        }

        // Trace of matrix product
        mu_trans(dir) = (tdm.transpose() * mu).trace();
    }

    return mu_trans;
}

double TransitionDensity::oscillator_strength(
    const Eigen::Vector3d& transition_dipole,
    double delta_e
) {
    if (delta_e <= 0.0) {
        throw std::invalid_argument(
            "TransitionDensity::oscillator_strength: "
            "Energy difference must be positive (excitation energy)"
        );
    }

    // Oscillator strength: f_IJ = (2/3) * ΔE * |μ_IJ|²
    // ΔE in Hartree, μ in atomic units
    double mu_squared = transition_dipole.squaredNorm();
    double f = (2.0 / 3.0) * delta_e * mu_squared;

    return f;
}

void TransitionDensity::natural_transition_orbitals(
    const Eigen::MatrixXd& tdm,
    Eigen::MatrixXd& hole_orbitals,
    Eigen::MatrixXd& particle_orbitals,
    Eigen::VectorXd& amplitudes
) {
    // Singular value decomposition: γ^IJ = U Σ V†
    // U = hole orbitals (where electron comes from)
    // V = particle orbitals (where electron goes to)
    // Σ = transition amplitudes
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        tdm, 
        Eigen::ComputeFullU | Eigen::ComputeFullV
    );

    hole_orbitals = svd.matrixU();
    particle_orbitals = svd.matrixV();
    amplitudes = svd.singularValues();

    // NTOs are sorted by decreasing amplitude
    // Typically, first 1-2 pairs capture most of excitation character
}

void TransitionDensity::attachment_detachment_density(
    const Eigen::MatrixXd& tdm,
    Eigen::VectorXd& attachment,
    Eigen::VectorXd& detachment
) {
    int n_orb = tdm.rows();
    
    // Attachment density: ρ_attach = diagonal of γ · γ†
    // Represents where electron is added
    Eigen::MatrixXd gamma_gamma_dag = tdm * tdm.transpose();
    attachment = gamma_gamma_dag.diagonal();

    // Detachment density: ρ_detach = diagonal of γ† · γ
    // Represents where electron is removed
    Eigen::MatrixXd gamma_dag_gamma = tdm.transpose() * tdm;
    detachment = gamma_dag_gamma.diagonal();
}

bool TransitionDensity::is_dipole_allowed(
    const Eigen::Vector3d& transition_dipole,
    double threshold
) {
    // Electric dipole selection rules:
    // - ΔS = 0 (spin conservation, not checked here)
    // - |μ_IJ| > 0 (non-zero transition dipole)
    
    double dipole_magnitude = transition_dipole.norm();
    return (dipole_magnitude > threshold);
}

double TransitionDensity::einstein_a_coefficient(
    const Eigen::Vector3d& transition_dipole,
    double delta_e
) {
    if (delta_e <= 0.0) {
        throw std::invalid_argument(
            "TransitionDensity::einstein_a_coefficient: "
            "Energy difference must be positive"
        );
    }

    // Einstein A coefficient for spontaneous emission:
    //   A_IJ = (2 * ω³ * |μ|²) / (3 * c³)
    // 
    // where:
    //   ω = transition frequency (equal to ΔE in atomic units, ħ=1)
    //   μ = transition dipole (atomic units)
    //   c = speed of light (atomic units = 137.036)
    //
    // Returns rate in a.u. (time⁻¹)
    // Convert to s⁻¹ by multiplying by 4.134×10¹⁶

    double omega = delta_e;  // ω = ΔE in a.u. (ħ=1)
    double omega_cubed = omega * omega * omega;
    double mu_squared = transition_dipole.squaredNorm();
    double c_cubed = SPEED_OF_LIGHT_AU * SPEED_OF_LIGHT_AU * SPEED_OF_LIGHT_AU;

    double A_coeff = (2.0 * omega_cubed * mu_squared) / (3.0 * c_cubed);

    return A_coeff;
}

} // namespace ci
} // namespace mshqc
