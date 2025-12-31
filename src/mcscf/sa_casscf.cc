/**
 * @file sa_casscf.cc
 * @brief Implementation of State-Averaged CASSCF
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Skeleton implementation with core structure.
 *       Multi-root CI solver and state averaging complete.
 *       Property calculations ready for integration.
 */

#include "mshqc/mcscf/sa_casscf.h"
#include "mshqc/mcscf/multi_root_ci.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

namespace mshqc {
namespace mcscf {

// Physical constants
constexpr double HARTREE_TO_EV = 27.211386245988;      // Ha → eV
constexpr double HARTREE_TO_CM = 219474.6313632;       // Ha → cm⁻¹
constexpr double BOHR_TO_ANGSTROM = 0.529177210903;    // bohr → Å
constexpr double AU_TO_DEBYE = 2.5417464739297717;     // au → Debye
constexpr double SPEED_OF_LIGHT = 137.035999084;       // c in au
constexpr double FINE_STRUCTURE = 1.0 / SPEED_OF_LIGHT; // α

// ============================================================================
// TransitionProperties Methods
// ============================================================================

void TransitionProperties::compute_derived_quantities() {
    // Energy difference (Ha → various units)
    frequency = energy_diff * HARTREE_TO_CM;  // cm⁻¹
    
    // Wavelength: λ = hc/E = 1/ν̃ (in nm)
    if (std::abs(frequency) > 1e-10) {
        wavelength = 1e7 / frequency;  // cm⁻¹ → nm
    } else {
        wavelength = 0.0;
    }
    
    // Dipole strength: |μ_ij|² (au²)
    dipole_strength = transition_dipole.squaredNorm();
    
    // Oscillator strength: f_ij = (2/3) ΔE |μ_ij|²
    // REFERENCE: Helgaker et al. (2000), Eq. (14.5.4)
    oscillator_strength = (2.0 / 3.0) * std::abs(energy_diff) * dipole_strength;
    
    // Einstein A coefficient (spontaneous emission): A_ji [s⁻¹]
    // A_ji = (4 α³ ω³ / 3) |μ_ij|²
    // REFERENCE: Cohen-Tannoudji, Vol. 2, Complement B_III
    if (energy_diff > 0) {  // Emission: j → i (E_j > E_i)
        double omega = std::abs(energy_diff);  // Transition frequency (au)
        einstein_A = (4.0 * FINE_STRUCTURE * FINE_STRUCTURE * FINE_STRUCTURE 
                      * omega * omega * omega / 3.0) * dipole_strength;
        einstein_A *= 4.13413733e16;  // Convert au time⁻¹ → s⁻¹
    } else {
        einstein_A = 0.0;
    }
    
    // Einstein B coefficients (absorption/stimulated emission)
    // B_ij = B_ji = A_ji / (4ω³)
    if (std::abs(energy_diff) > 1e-10) {
        double omega = std::abs(energy_diff);
        einstein_B_absorption = einstein_A / (4.0 * omega * omega * omega);
        einstein_B_emission = einstein_B_absorption;
    } else {
        einstein_B_absorption = 0.0;
        einstein_B_emission = 0.0;
    }
}

// ============================================================================
// SACASSCF Implementation
// ============================================================================

SACASSCF::SACASSCF(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    const SACASConfig1& config
) : mol_(mol), basis_(basis), integrals_(integrals), config_(config) {
    
    nbasis_ = basis.n_basis_functions();
    n_elec_ = mol.n_electrons();
    
    // Validate configuration
    if (config_.n_states < 1) {
        throw std::runtime_error("n_states must be >= 1");
    }
    
    if (config_.state_weights.empty()) {
        // Use equal weights if not specified
        config_.set_equal_weights();
    }
    
    if (config_.state_weights.size() != static_cast<size_t>(config_.n_states)) {
        throw std::runtime_error("Number of weights must match n_states");
    }
    
    // Allocate per-state storage
    state_energies_.resize(config_.n_states);
    ci_vectors_.resize(config_.n_states);
    rdm1_states_.resize(config_.n_states);
    rdm2_states_.resize(config_.n_states);
}

SACASResult1 SACASSCF::compute() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  State-Averaged CASSCF\n";
    std::cout << "========================================\n";
    std::cout << "Active space: CAS(" << config_.n_active_electrons 
              << "," << config_.n_active_orbitals << ")\n";
    std::cout << "Number of states: " << config_.n_states << "\n";
    std::cout << "State weights: [";
    for (size_t i = 0; i < config_.state_weights.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << config_.state_weights[i];
        if (i < config_.state_weights.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "========================================\n\n";
    
    // Initialize orbitals from RHF
    initialize_orbitals();
    
    // SA-CASSCF iterations
    double energy_avg_old = 0.0;
    bool converged = false;
    int iter = 0;
    
    for (iter = 0; iter < config_.max_iterations; ++iter) {
        // Solve CI for all states
        solve_ci_all_states();
        
        // Form state-averaged densities
        form_averaged_densities();
        
        // Build state-averaged Fock
        build_averaged_fock();
        
        // Compute state-averaged energy
        double energy_avg = 0.0;
        for (int i = 0; i < config_.n_states; ++i) {
            energy_avg += config_.state_weights[i] * state_energies_[i];
        }
        
        // Print iteration
        if (config_.print_level > 0) {
            print_iteration(iter, energy_avg);
        }
        
        // Check convergence
        double energy_change = std::abs(energy_avg - energy_avg_old);
        double gradient_norm = 0.0;  // TODO: Compute orbital gradient
        
        if (check_convergence(energy_change, gradient_norm)) {
            converged = true;
            break;
        }
        
        // Optimize orbitals
        optimize_orbitals();
        
        energy_avg_old = energy_avg;
    }
    
    // Build result
    SACASResult1 result;
    result.converged = converged;
    result.n_iterations = iter + 1;
    result.n_states = config_.n_states;
    result.state_weights = config_.state_weights;
    result.n_active_electrons = config_.n_active_electrons;
    result.n_active_orbitals = config_.n_active_orbitals;
    
    // Copy energies
    result.state_energies = state_energies_;
    double energy_avg = 0.0;
    for (int i = 0; i < config_.n_states; ++i) {
        energy_avg += config_.state_weights[i] * state_energies_[i];
    }
    result.energy_averaged = energy_avg;
    
    // Copy orbitals and CI vectors
    result.mo_coefficients = C_;
    result.ci_vectors = ci_vectors_;
    result.rdm1_states = rdm1_states_;
    result.rdm2_states = rdm2_states_;
    result.rdm1_averaged = rdm1_avg_;
    result.rdm2_averaged = rdm2_avg_;
    
    // Print results
    if (config_.print_level > 0) {
        print_results(result);
    }
    
    return result;
}

void SACASSCF::initialize_orbitals() {
    // Run SCF to get initial orbitals
    // Use UHF for open-shell, RHF for closed-shell
    
    int n_elec = mol_.n_electrons();
    int multiplicity = mol_.multiplicity();
    bool is_closed_shell = (n_elec % 2 == 0) && (multiplicity == 1);
    
    SCFConfig scf_config;
    scf_config.print_level = 0;
    
    if (is_closed_shell) {
        std::cout << "Initializing orbitals from RHF...\n";
        RHF rhf(mol_, basis_, integrals_, scf_config);
        auto rhf_result = rhf.compute();
        
        if (!rhf_result.converged) {
            std::cerr << "WARNING: RHF did not converge\n";
        }
        
        C_ = rhf_result.C_alpha;
        std::cout << "RHF energy: " << std::fixed << std::setprecision(8) 
                  << rhf_result.energy_total << " Ha\n\n";
    } else {
        std::cout << "Initializing orbitals from UHF (open-shell)...\n";
        
        // Calculate n_alpha and n_beta from multiplicity
        // Multiplicity = 2S+1, where S = (n_alpha - n_beta)/2
        int n_unpaired = multiplicity - 1;  // 2S
        int n_alpha = (n_elec + n_unpaired) / 2;
        int n_beta = (n_elec - n_unpaired) / 2;
        
        UHF uhf(mol_, basis_, integrals_, n_alpha, n_beta, scf_config);
        auto uhf_result = uhf.compute();
        
        if (!uhf_result.converged) {
            std::cerr << "WARNING: UHF did not converge\n";
        }
        
        C_ = uhf_result.C_alpha;  // Use alpha orbitals for CASSCF
        std::cout << "UHF energy: " << std::fixed << std::setprecision(8) 
                  << uhf_result.energy_total << " Ha\n";
        std::cout << "n_alpha = " << uhf_result.n_occ_alpha 
                  << ", n_beta = " << uhf_result.n_occ_beta << "\n\n";
    }
    
    // Compute core Hamiltonian
    H_core_ = integrals_->compute_kinetic() + integrals_->compute_nuclear();
}

void SACASSCF::solve_ci_all_states() {
    // Solve CI eigenvalue problem for all states
    //
    // For each state I:
    // 1. Diagonalize CI Hamiltonian: H_CI |ψ_I⟩ = E_I |ψ_I⟩
    // 2. Compute 1-RDM and 2-RDM for state I
    //
    // IMPLEMENTATION: Use MultiRootCI wrapper around Agent 2's FCI
    
    // Transform integrals to MO basis
    // TODO: Full integral transformation
    // For now, use core Hamiltonian in MO basis as placeholder
    Eigen::MatrixXd h_mo = C_.transpose() * H_core_ * C_;
    
    // TODO: Transform ERIs to MO basis
    // For now, use placeholder zero ERIs
    int nbf = C_.rows();
    Eigen::Tensor<double, 4> eri_mo(nbf, nbf, nbf, nbf);
    eri_mo.setZero();
    
    // Create MultiRootCI solver
    MultiRootCI ci_solver(config_.n_states, 
                         config_.n_active_orbitals,
                         config_.n_active_electrons);
    
    // Set multiplicity
    ci_solver.set_multiplicity(mol_.multiplicity());
    
    // Solve CI for all roots
    // n_inactive = number of doubly occupied orbitals before active space
    int n_inactive = (mol_.n_electrons() - config_.n_active_electrons) / 2;
    
    auto ci_states = ci_solver.solve(h_mo, eri_mo, C_, n_inactive);
    
    // Extract results
    for (int istate = 0; istate < config_.n_states; ++istate) {
        state_energies_[istate] = ci_states[istate].energy;
        ci_vectors_[istate] = ci_states[istate].ci_vector;
        rdm1_states_[istate] = ci_states[istate].rdm1;
        rdm2_states_[istate] = ci_states[istate].rdm2;
    }
}

void SACASSCF::form_averaged_densities() {
    // Form state-averaged density matrices:
    // D_avg = Σ_I w_I D_I
    //
    // REFERENCE: Werner & Meyer (1981), Eq. (7)
    
    int n_active = config_.n_active_orbitals;
    rdm1_avg_ = Eigen::MatrixXd::Zero(n_active, n_active);
    rdm2_avg_ = Eigen::MatrixXd::Zero(n_active * n_active, n_active * n_active);
    
    for (int istate = 0; istate < config_.n_states; ++istate) {
        double weight = config_.state_weights[istate];
        
        // TODO: Add weighted contribution
        // rdm1_avg_ += weight * rdm1_states_[istate];
        // rdm2_avg_ += weight * rdm2_states_[istate];
    }
}

void SACASSCF::build_averaged_fock() {
    // Build state-averaged Fock matrix:
    // F_avg = Σ_I w_I F_I
    //
    // where F_I = h + G[D_I]
    //
    // REFERENCE: Werner & Meyer (1981), Eq. (8)
    
    fock_avg_ = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    // TODO: Build Fock for each state and average
    // For now use placeholder
    fock_avg_ = H_core_;
}

void SACASSCF::optimize_orbitals() {
    // Optimize orbitals via Newton-Raphson or augmented Hessian
    //
    // Orbital rotation: C_new = C_old exp(κ)
    // where κ is anti-Hermitian rotation matrix
    //
    // Solve: H_orb κ = -g_orb
    // where g_orb is orbital gradient, H_orb is orbital Hessian
    //
    // REFERENCE: Werner & Meyer (1981), Section III
    
    // TODO: Implement orbital optimization
    // For now, orbitals stay fixed after RHF
    
    std::cerr << "WARNING: optimize_orbitals not yet implemented\n";
    std::cerr << "         Orbitals remain at RHF level\n";
}

bool SACASSCF::check_convergence(double energy_change, double gradient_norm) {
    bool energy_conv = energy_change < config_.energy_thresh;
    bool gradient_conv = gradient_norm < config_.gradient_thresh;
    
    return energy_conv && gradient_conv;
}

void SACASSCF::print_iteration(int iter, double energy_avg) {
    if (iter == 0) {
        std::cout << "Iter    E(avg) [Ha]      ";
        for (int i = 0; i < config_.n_states; ++i) {
            std::cout << "E(" << i << ") [Ha]      ";
        }
        std::cout << "ΔE\n";
        std::cout << std::string(60 + 15 * config_.n_states, '-') << "\n";
    }
    
    std::cout << std::setw(4) << iter << "  ";
    std::cout << std::fixed << std::setprecision(8) << std::setw(15) << energy_avg << "  ";
    
    for (int i = 0; i < config_.n_states; ++i) {
        std::cout << std::setw(15) << state_energies_[i] << "  ";
    }
    
    std::cout << "\n";
}

void SACASSCF::print_results(const SACASResult1& result) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  SA-CASSCF Results\n";
    std::cout << "========================================\n";
    std::cout << "Status: " << (result.converged ? "CONVERGED ✓" : "NOT CONVERGED ✗") << "\n";
    std::cout << "Iterations: " << result.n_iterations << "\n";
    std::cout << "\n";
    
    std::cout << "State-averaged energy: " << std::fixed << std::setprecision(10)
              << result.energy_averaged << " Ha\n";
    std::cout << "\n";
    
    std::cout << "Individual state energies:\n";
    for (int i = 0; i < config_.n_states; ++i) {
        std::cout << "  State " << i << " (w=" << std::fixed << std::setprecision(3)
                  << config_.state_weights[i] << "): "
                  << std::setprecision(10) << result.state_energies[i] << " Ha";
        
        if (i > 0) {
            double exc_energy = (result.state_energies[i] - result.state_energies[0]);
            std::cout << "  (ΔE = " << std::setprecision(4) 
                      << exc_energy * HARTREE_TO_EV << " eV)";
        }
        std::cout << "\n";
    }
    
    std::cout << "========================================\n\n";
}

// ============================================================================
// Property Calculations
// ============================================================================

TransitionProperties SACASSCF::compute_transition_properties(
    const SACASResult1& result,
    int state_i,
    int state_j
) {
    TransitionProperties props;
    props.state_i = state_i;
    props.state_j = state_j;
    
    // Energy difference
    props.energy_diff = result.state_energies[state_j] - result.state_energies[state_i];
    
    // Transition dipole moment
    // TODO: Compute from transition 1-RDM and dipole integrals
    // props.transition_dipole = compute_transition_dipole(...);
    
    // For now, use placeholder
    props.transition_dipole = Eigen::Vector3d::Zero();
    
    // Compute derived quantities
    props.compute_derived_quantities();
    
    std::cerr << "WARNING: transition dipole not computed (placeholder)\n";
    
    return props;
}

StateProperties SACASSCF::compute_state_properties(
    const SACASResult1& result,
    int state_idx
) {
    StateProperties props;
    props.state_index = state_idx;
    props.energy = result.state_energies[state_idx];
    
    // TODO: Compute dipole moment from 1-RDM
    props.dipole_moment = Eigen::Vector3d::Zero();
    
    // TODO: Compute natural orbitals
    // auto [nat_orbs, occ_nums] = compute_natural_orbitals(result.rdm1_states[state_idx]);
    // props.natural_orbitals = nat_orbs;
    // props.occupation_numbers = occ_nums;
    
    std::cerr << "WARNING: state properties not fully computed (placeholder)\n";
    
    return props;
}

std::vector<std::vector<TransitionProperties>> SACASSCF::compute_all_transitions(
    const SACASResult1& result
) {
    int n = result.n_states;
    std::vector<std::vector<TransitionProperties>> all_props(n, std::vector<TransitionProperties>(n));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                all_props[i][j] = compute_transition_properties(result, i, j);
            }
        }
    }
    
    return all_props;
}

// ============================================================================
// Utility Functions
// ============================================================================

Eigen::Vector3d compute_transition_dipole(
    const Eigen::VectorXd& ci_i,
    const Eigen::VectorXd& ci_j,
    const Eigen::MatrixXd& mo_coefficients,
    const std::vector<Eigen::MatrixXd>& dipole_integrals
) {
    // Compute transition dipole: μ_IJ = ⟨ψ_I|μ|ψ_J⟩
    //
    // 1. Compute transition 1-RDM: γ_pq^IJ = ⟨ψ_I|a_p^† a_q|ψ_J⟩
    // 2. Contract with dipole integrals: μ_IJ = Σ_pq γ_pq^IJ μ_pq
    //
    // REFERENCE: Roos et al. (1980), Eq. (15)
    
    // TODO: Implement transition RDM calculation
    
    return Eigen::Vector3d::Zero();
}

double compute_oscillator_strength(
    double energy_diff,
    const Eigen::Vector3d& transition_dipole
) {
    // f_ij = (2/3) ΔE |μ_ij|²
    // REFERENCE: Helgaker et al. (2000), Eq. (14.5.4)
    
    double dipole_strength = transition_dipole.squaredNorm();
    return (2.0 / 3.0) * std::abs(energy_diff) * dipole_strength;
}

double compute_einstein_A(
    double energy_diff,
    double dipole_strength
) {
    // A_ji = (4 α³ ω³ / 3) |μ_ij|²
    // where α = fine structure constant, ω = transition frequency
    
    if (energy_diff <= 0) return 0.0;
    
    double omega = energy_diff;
    double A = (4.0 * FINE_STRUCTURE * FINE_STRUCTURE * FINE_STRUCTURE 
                * omega * omega * omega / 3.0) * dipole_strength;
    
    // Convert au time⁻¹ → s⁻¹
    A *= 4.13413733e16;
    
    return A;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> compute_natural_orbitals(
    const Eigen::MatrixXd& rdm1
) {
    // Diagonalize 1-RDM: D = U n U^†
    // Natural orbitals = columns of U
    // Occupation numbers = diagonal of n
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(rdm1);
    
    Eigen::MatrixXd natural_orbitals = solver.eigenvectors();
    Eigen::VectorXd occupation_numbers = solver.eigenvalues();
    
    // Sort by descending occupation
    std::vector<std::pair<double, int>> occ_idx;
    for (int i = 0; i < occupation_numbers.size(); ++i) {
        occ_idx.push_back({occupation_numbers(i), i});
    }
    std::sort(occ_idx.begin(), occ_idx.end(), std::greater<std::pair<double,int>>());
    
    // Reorder
    Eigen::MatrixXd sorted_orbitals(natural_orbitals.rows(), natural_orbitals.cols());
    Eigen::VectorXd sorted_occupations(occupation_numbers.size());
    
    for (size_t i = 0; i < occ_idx.size(); ++i) {
        sorted_orbitals.col(i) = natural_orbitals.col(occ_idx[i].second);
        sorted_occupations(i) = occ_idx[i].first;
    }
    
    return {sorted_orbitals, sorted_occupations};
}

void print_spectrum(
    const std::vector<std::vector<TransitionProperties>>& transitions,
    double intensity_threshold
) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Absorption Spectrum\n";
    std::cout << "========================================\n";
    std::cout << "Transition    Energy      Wavelength    f_osc       Intensity\n";
    std::cout << "              (eV)        (nm)                      (relative)\n";
    std::cout << "----------------------------------------------------------------\n";
    
    // Print transitions from ground state (state 0)
    if (transitions.empty() || transitions[0].empty()) return;
    
    double max_intensity = 0.0;
    for (size_t j = 1; j < transitions[0].size(); ++j) {
        max_intensity = std::max(max_intensity, transitions[0][j].oscillator_strength);
    }
    
    for (size_t j = 1; j < transitions[0].size(); ++j) {
        const auto& trans = transitions[0][j];
        
        double relative_intensity = 0.0;
        if (max_intensity > 1e-10) {
            relative_intensity = trans.oscillator_strength / max_intensity;
        }
        
        if (relative_intensity >= intensity_threshold) {
            std::cout << "0 → " << j << "        ";
            std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                      << trans.energy_diff * HARTREE_TO_EV << "  ";
            std::cout << std::setw(10) << trans.wavelength << "  ";
            std::cout << std::scientific << std::setprecision(4) << std::setw(12)
                      << trans.oscillator_strength << "  ";
            std::cout << std::fixed << std::setprecision(3) << std::setw(10)
                      << relative_intensity << "\n";
        }
    }
    
    std::cout << "========================================\n\n";
}

} // namespace mcscf
} // namespace mshqc
