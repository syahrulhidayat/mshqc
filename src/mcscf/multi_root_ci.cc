/**
 * @file multi_root_ci.cc
 * @brief Multi-root CI solver implementation
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * ORIGINALITY:
 * This implementation integrates 's FCI module for SA-CASSCF.
 * RDM algorithms derived from Helgaker et al. (2000) Chapter 11.
 * No code copied from external quantum chemistry packages.
 */

#include "mshqc/mcscf/multi_root_ci.h"
#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace mshqc {
namespace mcscf {

MultiRootCI::MultiRootCI(int n_states, int n_active_orb, int n_active_elec)
    : n_states_(n_states),
      n_active_orb_(n_active_orb),
      n_active_elec_(n_active_elec),
      multiplicity_(1),  // Default: singlet
      n_alpha_(0),
      n_beta_(0) {
    
    if (n_states < 1) {
        throw std::runtime_error("n_states must be >= 1");
    }
    
    if (n_active_orb < 1) {
        throw std::runtime_error("n_active_orb must be >= 1");
    }
    
    if (n_active_elec < 0 || n_active_elec > 2 * n_active_orb) {
        throw std::runtime_error("Invalid n_active_elec");
    }
    
    // Determine spin configuration
    determine_spin_config();
}

void MultiRootCI::set_multiplicity(int multiplicity) {
    multiplicity_ = multiplicity;
    determine_spin_config();
}

void MultiRootCI::determine_spin_config() {
    // Multiplicity = 2S + 1
    // S = (n_alpha - n_beta) / 2
    // n_alpha + n_beta = n_elec
    //
    // Solving: n_alpha = (n_elec + 2S) / 2
    //          n_beta = (n_elec - 2S) / 2
    
    int two_S = multiplicity_ - 1;  // 2S
    n_alpha_ = (n_active_elec_ + two_S) / 2;
    n_beta_ = (n_active_elec_ - two_S) / 2;
    
    if (n_alpha_ < 0 || n_beta_ < 0) {
        throw std::runtime_error("Invalid multiplicity for given n_electrons");
    }
    
    if (n_alpha_ > n_active_orb_ || n_beta_ > n_active_orb_) {
        throw std::runtime_error("Too many electrons for active space");
    }
}

size_t MultiRootCI::estimate_n_determinants() const {
    // Use FCI binomial estimate
    return ci::fci_determinant_count(n_active_orb_, n_alpha_, n_beta_);
}

std::vector<CIState> MultiRootCI::solve(
    const Eigen::MatrixXd& h_mo,
    const Eigen::Tensor<double, 4>& eri_mo,
    const Eigen::MatrixXd& C_mo,
    int n_inactive
) {
    std::cout << "\n=== Multi-Root CI Solver ===\n";
    std::cout << "Active space: (" << n_active_elec_ << "," << n_active_orb_ << ")\n";
    std::cout << "Multiplicity: " << multiplicity_ 
              << " (n_alpha=" << n_alpha_ << ", n_beta=" << n_beta_ << ")\n";
    std::cout << "Number of states: " << n_states_ << "\n";
    
    // Extract active space integrals
    auto h_active = extract_active_h(h_mo, n_inactive, n_active_orb_);
    auto eri_active = extract_active_eri(eri_mo, n_inactive, n_active_orb_);
    
    // Package as CIIntegrals for 's FCI
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = h_active;  // Same for alpha/beta in RHF
    ci_ints.h_beta = h_active;
    ci_ints.eri_aaaa = eri_active;  // Alpha-alpha
    ci_ints.eri_bbbb = eri_active;  // Beta-beta (same in RHF)
    ci_ints.eri_aabb = eri_active;  // Alpha-beta
    ci_ints.e_nuc = 0.0;  // Not needed for active space
    ci_ints.use_fock = false;  // Using bare Hamiltonian
    
    // Estimate determinant count
    size_t n_dets = estimate_n_determinants();
    std::cout << "Estimated determinants: " << n_dets << "\n";
    
    // Run FCI with multiple roots
    ci::FCI fci(ci_ints, n_active_orb_, n_alpha_, n_beta_, n_states_);
    auto fci_result = fci.compute();
    
    std::cout << "FCI converged: " << (fci_result.converged ? "Yes" : "No") << "\n";
    std::cout << "Actual determinants: " << fci_result.n_determinants << "\n\n";
    
    // Build CIState for each root
    std::vector<CIState> states;
    
    for (int i = 0; i < n_states_; ++i) {
        CIState state;
        state.determinants = fci_result.determinants;
        state.converged = fci_result.converged;
        state.iterations = fci_result.iterations;
        
        if (i == 0) {
            // Ground state
            state.energy = fci_result.e_fci;
            state.ci_vector = fci_result.coefficients;
        } else {
            // Excited state
            if (i - 1 < static_cast<int>(fci_result.excited_energies.size())) {
                state.energy = fci_result.excited_energies[i - 1];
                state.ci_vector = fci_result.excited_states[i - 1];
            } else {
                throw std::runtime_error("FCI did not return enough excited states");
            }
        }
        
        // Compute RDMs
        state.rdm1 = compute_rdm1(state.ci_vector, state.determinants, n_active_orb_);
        state.rdm2 = compute_rdm2(state.ci_vector, state.determinants, n_active_orb_);
        
        std::cout << "State " << i << ": E = " << std::fixed << std::setprecision(8) 
                  << state.energy << " Ha\n";
        
        states.push_back(state);
    }
    
    std::cout << "===========================\n\n";
    
    return states;
}

Eigen::MatrixXd MultiRootCI::extract_active_h(
    const Eigen::MatrixXd& h_mo,
    int start,
    int size
) {
    // Extract block [start:start+size, start:start+size]
    return h_mo.block(start, start, size, size);
}

Eigen::Tensor<double, 4> MultiRootCI::extract_active_eri(
    const Eigen::Tensor<double, 4>& eri_mo,
    int start,
    int size
) {
    // Extract active space block
    Eigen::Tensor<double, 4> eri_active(size, size, size, size);
    
    for (int p = 0; p < size; ++p) {
        for (int q = 0; q < size; ++q) {
            for (int r = 0; r < size; ++r) {
                for (int s = 0; s < size; ++s) {
                    eri_active(p, q, r, s) = eri_mo(start + p, start + q, 
                                                     start + r, start + s);
                }
            }
        }
    }
    
    return eri_active;
}

Eigen::MatrixXd MultiRootCI::compute_rdm1(
    const Eigen::VectorXd& ci_vector,
    const std::vector<ci::Determinant>& dets,
    int n_orb
) {
    // 1-RDM: γ_pq = ⟨ψ|a_p† a_q|ψ⟩
    //
    // ALGORITHM:
    // For all determinant pairs I, J:
    //   Check if J differs from I by single excitation p→q
    //   If yes: γ_pq += c_I * c_J * phase
    //
    // REFERENCE: Helgaker et al. (2000), Eq. (11.7.1)
    
    Eigen::MatrixXd rdm1 = Eigen::MatrixXd::Zero(n_orb, n_orb);
    
    int n_dets = dets.size();
    
    // Loop over all determinant pairs
    for (int i = 0; i < n_dets; ++i) {
        for (int j = 0; j < n_dets; ++j) {
            int exc_level = excitation_level(dets[i], dets[j]);
            
            if (exc_level == 0) {
                // Diagonal: ⟨I|a_p† a_p|I⟩ = occupation of p in I
                const auto& det = dets[i];
                double c_i = ci_vector(i);
                
                // Alpha orbitals
                for (int p = 0; p < n_orb; ++p) {
                    if (det.is_occupied(p, true)) {  // true = alpha
                        rdm1(p, p) += c_i * c_i;
                    }
                }
                
                // Beta orbitals
                for (int p = 0; p < n_orb; ++p) {
                    if (det.is_occupied(p, false)) {  // false = beta
                        rdm1(p, p) += c_i * c_i;
                    }
                }
                
            } else if (exc_level == 1) {
                // Single excitation: get indices and phase
                int p_alpha = -1, q_alpha = -1;
                int p_beta = -1, q_beta = -1;
                
                int phase = get_single_excitation_indices(
                    dets[i], dets[j], p_alpha, q_alpha, p_beta, q_beta
                );
                
                double contribution = ci_vector(i) * ci_vector(j) * phase;
                
                // Alpha contribution
                if (p_alpha >= 0 && q_alpha >= 0) {
                    rdm1(q_alpha, p_alpha) += contribution;
                }
                
                // Beta contribution
                if (p_beta >= 0 && q_beta >= 0) {
                    rdm1(q_beta, p_beta) += contribution;
                }
            }
            // exc_level > 1: contributes zero
        }
    }
    
    return rdm1;
}

Eigen::MatrixXd MultiRootCI::compute_rdm2(
    const Eigen::VectorXd& ci_vector,
    const std::vector<ci::Determinant>& dets,
    int n_orb
) {
    // 2-RDM: Γ_pqrs = ⟨ψ|a_p† a_q† a_s a_r|ψ⟩
    //
    // SIMPLIFIED: For SA-CASSCF, we mainly need trace properties
    // Full 2-RDM is expensive (n⁴ elements)
    //
    // TODO: Implement full 2-RDM when needed for orbital optimization
    // For now, return placeholder
    
    int n_elem = n_orb * n_orb;
    Eigen::MatrixXd rdm2 = Eigen::MatrixXd::Zero(n_elem, n_elem);
    
    // Placeholder: Use diagonal from 1-RDM for consistency
    // Real implementation would loop over double excitations
    
    std::cerr << "WARNING: compute_rdm2 using placeholder\n";
    std::cerr << "         Full 2-RDM not yet implemented\n";
    
    return rdm2;
}

int MultiRootCI::excitation_level(
    const ci::Determinant& det_i,
    const ci::Determinant& det_j
) {
    // Count differing orbitals
    int diff_alpha = 0;
    int diff_beta = 0;
    
    for (int p = 0; p < n_active_orb_; ++p) {
        if (det_i.is_occupied(p, true) != det_j.is_occupied(p, true)) {
            diff_alpha++;
        }
        if (det_i.is_occupied(p, false) != det_j.is_occupied(p, false)) {
            diff_beta++;
        }
    }
    
    // Excitation level = (diff_alpha + diff_beta) / 2
    return (diff_alpha + diff_beta) / 2;
}

int MultiRootCI::get_single_excitation_indices(
    const ci::Determinant& det_i,
    const ci::Determinant& det_j,
    int& p_alpha, int& q_alpha,
    int& p_beta, int& q_beta
) {
    // Find orbitals that differ
    // p = destroyed (in I but not J)
    // q = created (in J but not I)
    
    p_alpha = -1; q_alpha = -1;
    p_beta = -1; q_beta = -1;
    
    // Alpha spin
    for (int p = 0; p < n_active_orb_; ++p) {
        if (det_i.is_occupied(p, true) && !det_j.is_occupied(p, true)) {
            p_alpha = p;  // Destroyed
        }
        if (!det_i.is_occupied(p, true) && det_j.is_occupied(p, true)) {
            q_alpha = p;  // Created
        }
    }
    
    // Beta spin
    for (int p = 0; p < n_active_orb_; ++p) {
        if (det_i.is_occupied(p, false) && !det_j.is_occupied(p, false)) {
            p_beta = p;
        }
        if (!det_i.is_occupied(p, false) && det_j.is_occupied(p, false)) {
            q_beta = p;
        }
    }
    
    // Compute phase (simplified - assume +1 for now)
    // TODO: Compute actual Slater-Condon phase
    int phase = 1;
    
    return phase;
}

} // namespace mcscf
} // namespace mshqc
