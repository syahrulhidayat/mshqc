/**
 * @file scf.h
 * @brief Restricted Open-shell Hartree-Fock (ROHF) implementation
 * 
 * THEORY:
 * The ROHF method treats systems with unpaired electrons where
 * spatial orbitals are doubly occupied (closed-shell) or singly
 * occupied (open-shell), with all electrons in the same spatial orbital
 * having parallel spins.
 * 
 * REFERENCES:
 * [1] Roothaan, C. C. J., Rev. Mod. Phys. 23, 69 (1951) - ROHF theory
 * [2] Szabo & Ostlund, "Modern Quantum Chemistry" (1996), Chapter 3
 * [3] Helgaker et al., "Molecular Electronic-Structure Theory" (2000), Ch. 14
 */

#ifndef MSHQC_SCF_H
#define MSHQC_SCF_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <memory>

namespace mshqc {

// ============================================================================
// SCF Configuration
// ============================================================================

/**
 * @brief SCF convergence and algorithm parameters
 */
struct SCFConfig {
    /// Maximum number of SCF iterations
    int max_iterations = 100;
    
    /// Energy convergence threshold (Ha)
    double energy_threshold = 1e-8;
    
    /// Density convergence threshold
    double density_threshold = 1e-6;
    
    /// DIIS convergence threshold (enable DIIS when error < this)
    double diis_threshold = 1e-2;
    
    /// Maximum DIIS vectors to store
    int diis_max_vectors = 8;
    
    /// Print level (0=minimal, 1=normal, 2=verbose)
    int print_level = 1;
    
    /// Level shift parameter (Ha) for difficult convergence
    double level_shift = 0.0;
};

// ============================================================================
// SCF Results
// ============================================================================

/**
 * @brief Results from SCF calculation
 */
struct SCFResult {
    /// Total electronic energy (Ha)
    double energy_electronic;
    
    /// Nuclear repulsion energy (Ha)
    double energy_nuclear;
    
    /// Total energy (Ha)
    double energy_total;
    
    /// Alpha orbital energies (Ha)
    Eigen::VectorXd orbital_energies_alpha;
    
    /// Beta orbital energies (Ha)
    Eigen::VectorXd orbital_energies_beta;
    
    /// Alpha MO coefficients (nbasis × nmo)
    Eigen::MatrixXd C_alpha;
    
    /// Beta MO coefficients (nbasis × nmo)
    Eigen::MatrixXd C_beta;
    
    /// Alpha density matrix (nbasis × nbasis)
    Eigen::MatrixXd P_alpha;
    
    /// Beta density matrix (nbasis × nbasis)
    Eigen::MatrixXd P_beta;
    
    /// Alpha Fock matrix (AO basis)
    Eigen::MatrixXd F_alpha;
    
    /// Beta Fock matrix (AO basis)
    Eigen::MatrixXd F_beta;
    
    /// Number of iterations to convergence
    int iterations;
    
    /// Whether SCF converged
    bool converged;
    
    /// Final energy gradient norm
    double gradient_norm;
    
    /// Number of occupied alpha orbitals
    int n_occ_alpha;
    
    /// Number of occupied beta orbitals
    int n_occ_beta;
};

// ============================================================================
// DIIS Convergence Accelerator
// ============================================================================

/**
 * @brief Direct Inversion in the Iterative Subspace (DIIS)
 * 
 * REFERENCE: Pulay, P., Chem. Phys. Lett. 73, 393 (1980), Eq. (11)
 * REFERENCE: Pulay, P., J. Comp. Chem. 3, 556 (1982) - DIIS for SCF
 * 
 * Accelerates SCF by extrapolating Fock matrix from history:
 * Find coefficients {c_i} minimizing ||Σ c_i e_i||² subject to Σ c_i = 1
 * where e_i = [F_i, P_i] = F_i*P_i*S - S*P_i*F_i (commutator in AO basis)
 * 
 * Solve linear system with B_ij = Tr(e_i^† e_j), then F_new = Σ c_i F_i
 */
class DIIS {
public:
    /**
     * @brief Construct DIIS extrapolator
     * @param max_vectors Maximum number of vectors to store
     */
    explicit DIIS(int max_vectors = 8);
    
    /**
     * @brief Add Fock matrix and error vector to DIIS history
     * @param F Fock matrix
     * @param error Error vector (F*P*S - S*P*F)
     */
    void add_iteration(const Eigen::MatrixXd& F, const Eigen::MatrixXd& error);
    
    /**
     * @brief Extrapolate Fock matrix using DIIS
     * @return Extrapolated Fock matrix (F_new = Σ c_i F_i)
     */
    Eigen::MatrixXd extrapolate();
    
    /**
     * @brief Clear DIIS history
     */
    void clear();
    
    /**
     * @brief Get number of stored vectors
     */
    size_t size() const { return fock_matrices_.size(); }
    
    /**
     * @brief Check if DIIS has enough vectors for extrapolation
     */
    bool can_extrapolate() const { return size() >= 2; }
    
private:
    int max_vectors_;  ///< Maximum number of vectors to store
    std::vector<Eigen::MatrixXd> fock_matrices_;  ///< History of Fock matrices
    std::vector<Eigen::MatrixXd> error_vectors_;  ///< History of error vectors
    
    /**
     * @brief Build DIIS B matrix
     * B_ij = Tr(e_i^† e_j)
     */
    Eigen::MatrixXd build_B_matrix() const;
};

// ============================================================================
// RHF SCF Class
// ============================================================================

/**
 * @brief Restricted Hartree-Fock for closed-shell systems
 * 
 * REFERENCES:
 * Roothaan, C. C. J., Rev. Mod. Phys. 23, 69 (1951), Eq. (1)-(3)
 * Szabo & Ostlund (1996), Section 3.4, pp. 138-146
 * 
 * For N electrons (N even), N/2 doubly-occupied orbitals:
 * - All electrons occupy same spatial orbitals with opposite spins
 * - Single Fock matrix: F = H + 2J - K
 * - Simplest HF case: C^α = C^β, ε^α = ε^β
 * 
 * Energy: E = 2 Σ_i^{N/2} h_ii + Σ_ij^{N/2} (2<ij|ij> - <ij|ji>)
 */
class RHF {
public:
    /**
     * @brief Construct RHF calculator
     * @param mol Molecule
     * @param basis Basis set
     * @param integrals Integral engine
     * @param config SCF configuration parameters
     */
    RHF(const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SCFConfig& config = SCFConfig());
    
    /**
     * @brief Run RHF SCF calculation
     * @return SCF results
     */
    SCFResult compute();
    
    /**
     * @brief Get current energy
     */
    double energy() const { return energy_; }
    
    /**
     * @brief Get number of basis functions
     */
    size_t nbasis() const { return nbasis_; }
    
    /**
     * @brief Get number of occupied orbitals
     */
    int n_occ() const { return n_occ_; }
    
private:
    // Molecule and basis
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    size_t nbasis_;
    
    // Electron configuration
    int n_occ_;  ///< Number of doubly-occupied orbitals (N_elec/2)
    
    // Configuration
    SCFConfig config_;
    
    // Integrals
    Eigen::MatrixXd S_;   ///< Overlap matrix
    Eigen::MatrixXd H_;   ///< Core Hamiltonian
    Eigen::MatrixXd X_;   ///< Orthogonalization matrix (S^{-1/2})
    
    // SCF quantities
    Eigen::MatrixXd C_;    ///< MO coefficients
    Eigen::MatrixXd P_;    ///< Density matrix
    Eigen::MatrixXd F_;    ///< Fock matrix
    Eigen::VectorXd eps_;  ///< Orbital energies
    
    // Energy
    double energy_;
    double energy_old_;
    
    // DIIS
    DIIS diis_;
    
    /**
     * @brief Initialize integrals and orthogonalization
     */
    void init_integrals();
    
    /**
     * @brief Form initial guess (core Hamiltonian)
     * REFERENCE: Szabo & Ostlund (1996), Section 3.4.4
     */
    void initial_guess();
    
    /**
     * @brief Build Fock matrix F = H + G
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.154)
     * F_μν = H_μν + Σ_λσ P_λσ [(μν|λσ) - 0.5(μλ|νσ)]
     */
    void build_fock();
    
    /**
     * @brief Build density matrix P = 2 Σ_i^occ C_i C_i^T
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.145)
     */
    Eigen::MatrixXd build_density();
    
    /**
     * @brief Compute electronic energy
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.184)
     * E = Σ_μν P_μν (H_μν + F_μν) / 2
     */
    double compute_energy();
    
    /**
     * @brief Solve Fock equation F C = S C ε
     * REFERENCE: Szabo & Ostlund (1996), Section 3.4.5
     */
    void solve_fock();
    
    /**
     * @brief Check SCF convergence
     */
    bool check_convergence();
    
    /**
     * @brief Print iteration info
     */
    void print_iter(int iter, double de, double dp);
    
    /**
     * @brief Print final results
     */
    void print_final(const SCFResult& result);
};

// ============================================================================
// ROHF SCF Class
// ============================================================================

/**
 * @brief Restricted Open-Shell Hartree-Fock implementation
 * 
 * REFERENCES:
 * Roothaan, C. C. J., Rev. Mod. Phys. 32, 179 (1960), Eq. (5.10)-(5.12)
 * Szabo & Ostlund (1996), Section 3.8, pp. 135-141
 * 
 * For N_α alpha and N_β beta electrons (N_α > N_β):
 * - Doubly occupied: i = 1,...,N_β
 * - Singly occupied: a = N_β+1,...,N_α  
 * - Virtual: p = N_α+1,...,nbasis
 * 
 * Fock matrices:
 * F^α = H + G^closed + G^open,α where G^open,α has extra exchange
 * F^β = H + G^closed + G^open,β where G^open,β is pure Coulomb
 * 
 * Energy: E = ½ Tr[(P^α + P^β)H + P^α F^α + P^β F^β]
 */
class ROHF {
public:
    /**
     * @brief Construct ROHF calculator
     * @param mol Molecule
     * @param basis Basis set
     * @param n_alpha Number of alpha electrons
     * @param n_beta Number of beta electrons
     * @param config SCF configuration parameters
     */
    ROHF(const Molecule& mol,
         const BasisSet& basis,
         int n_alpha,
         int n_beta,
         const SCFConfig& config = SCFConfig());
    
    /**
     * @brief Run ROHF SCF calculation
     * @return SCF results
     * 
     * Standard SCF loop: Guess → Build F → DIIS → Solve FC=SCε → Update P → Check convergence
     */
    SCFResult run();
    
    /**
     * @brief Get current energy
     */
    double energy() const { return energy_; }
    
    /**
     * @brief Get number of basis functions
     */
    size_t nbasis() const { return nbasis_; }
    
    /**
     * @brief Get number of alpha electrons
     */
    int n_alpha() const { return n_alpha_; }
    
    /**
     * @brief Get number of beta electrons
     */
    int n_beta() const { return n_beta_; }
    
private:
    // Molecule and basis
    const Molecule& mol_;
    const BasisSet& basis_;
    size_t nbasis_;
    
    // Electron configuration
    int n_alpha_;  ///< Number of alpha electrons
    int n_beta_;   ///< Number of beta electrons
    
    // Configuration
    SCFConfig config_;
    
    // Integrals
    std::unique_ptr<IntegralEngine> integrals_;
    Eigen::MatrixXd S_;   ///< Overlap matrix
    Eigen::MatrixXd H_;   ///< Core Hamiltonian
    Eigen::Tensor<double, 4> ERI_;  ///< Electron repulsion integrals
    
    // SCF quantities
    Eigen::MatrixXd C_alpha_;  ///< Alpha MO coefficients
    Eigen::MatrixXd C_beta_;   ///< Beta MO coefficients
    Eigen::MatrixXd P_alpha_;  ///< Alpha density matrix
    Eigen::MatrixXd P_beta_;   ///< Beta density matrix
    Eigen::MatrixXd F_alpha_;  ///< Alpha Fock matrix
    Eigen::MatrixXd F_beta_;   ///< Beta Fock matrix
    Eigen::VectorXd eps_alpha_;  ///< Alpha orbital energies
    Eigen::VectorXd eps_beta_;   ///< Beta orbital energies
    
    // Energy
    double energy_;
    double energy_old_;
    
    /**
     * @brief Initialize integral engine and compute integrals
     */
    void initialize_integrals();
    
    /**
     * @brief Form initial guess for density matrices
     * 
     * REFERENCE: Szabo & Ostlund (1996), Section 3.4.4, p. 143
     * Core Hamiltonian guess: Diagonalize H → Occupy lowest orbitals → Build P
     */
    void initial_guess();
    
    /**
     * @brief Build Fock matrices for alpha and beta spins
     * 
     * REFERENCE: Roothaan, Rev. Mod. Phys. 32, 179 (1960), Eq. (5.10)-(5.12)
     * F = H + G^closed + G^open where G includes Coulomb (J) and exchange (K)
     */
    void build_fock();
    
    /**
     * @brief Build density matrix from MO coefficients
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.145), p. 139
     * P_μν = Σ_i^occ C_μi C_νi
     */
    Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ);
    
    /**
     * @brief Compute electronic energy from densities and Fock matrices
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (3.184), p. 150
     * E = ½ Tr[(P^α + P^β)H + P^α F^α + P^β F^β]
     */
    double compute_energy();
    
    /**
     * @brief Solve generalized eigenvalue problem F C = S C ε
     * 
     * REFERENCE: Szabo & Ostlund (1996), Section 3.4.5, pp. 143-145
     * Orthogonalize (X = S^-1/2) → Transform (F' = X^T F X) → Diagonalize → Back-transform
     */
    void solve_fock(const Eigen::MatrixXd& F,
                    Eigen::MatrixXd& C,
                    Eigen::VectorXd& eps);
    
    /**
     * @brief Check SCF convergence
     * @return true if converged (|ΔE| and ||ΔP|| below thresholds)
     */
    bool check_convergence();
    
    /**
     * @brief Print SCF iteration information
     */
    void print_iteration(int iter, double dE, double dP);
    
    /**
     * @brief Print final SCF results
     */
    void print_results(const SCFResult& result);
    
    /**
     * @brief Compute single ERI element (pq|rs)
     * 
     * Placeholder for on-the-fly ERI computation.
     * For now returns 0.0 - will be properly implemented.
     */
    double compute_eri_element(size_t p, size_t q, size_t r, size_t s);
};

// ============================================================================
// UHF SCF Class  
// ============================================================================

/**
 * Unrestricted Hartree-Fock for open-shell systems
 * 
 * REFERENCE: Pople & Nesbet (1954), J. Chem. Phys. 22, 571
 * REFERENCE: Szabo & Ostlund (1996), Section 3.8.5, pp. 108-110
 * 
 * Different from ROHF: α and β have DIFFERENT spatial orbitals
 * C^α ≠ C^β, ε^α ≠ ε^β → easier MP2 but spin contamination
 * 
 * Fock build:
 * F^α_μν = H_μν + Σ_λσ [(P^α + P^β)_λσ (μν|λσ) - P^α_λσ (μλ|νσ)]
 * F^β_μν = H_μν + Σ_λσ [(P^α + P^β)_λσ (μν|λσ) - P^β_λσ (μλ|νσ)]
 */
class UHF {
public:
    UHF(const Molecule& mol,
        const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        int n_alpha,
        int n_beta,
        const SCFConfig& config = SCFConfig());
    
    // Main compute
    SCFResult compute();
    
    // Spin contamination check
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.199)
    // <S²> = S(S+1) + N_β - Σ_ij |⟨φ^α_i|φ^β_j⟩|²
    double compute_s_squared(const SCFResult& result);
    
private:
    // System
    Molecule mol_;
    BasisSet basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFConfig config_;
    
    // Dimensions
    size_t nbasis_;
    int n_alpha_;  // # α electrons
    int n_beta_;   // # β electrons
    
    // Matrices (separate for α and β)
    Eigen::MatrixXd S_;    // overlap
    Eigen::MatrixXd H_;    // core Hamiltonian
    Eigen::MatrixXd X_;    // orthogonalization
    Eigen::MatrixXd C_a_;  // α MO coeffs
    Eigen::MatrixXd C_b_;  // β MO coeffs  
    Eigen::MatrixXd P_a_;  // α density
    Eigen::MatrixXd P_b_;  // β density
    Eigen::MatrixXd F_a_;  // α Fock
    Eigen::MatrixXd F_b_;  // β Fock
    Eigen::VectorXd eps_a_;  // α orbital energies
    Eigen::VectorXd eps_b_;  // β orbital energies
    
    double energy_, energy_old_;
    
    // Setup
    void init_integrals();
    void initial_guess();
    
    // Core UHF routines
    // Build separate Fock for each spin
    void build_fock();
    
    // Density from C: P_μν = Σ_i C_μi C_νi
    Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n);
    
    // Energy: E = Tr[P_tot H] + ½Tr[P^α F^α + P^β F^β]
    double compute_energy();
    
    // Solve FC = SCε (separately for α and β)
    void solve_fock(const Eigen::MatrixXd& F, 
                    Eigen::MatrixXd& C,
                    Eigen::VectorXd& eps);
    
    bool check_convergence();
    void print_iter(int iter, double de, double dp);
    void print_final(const SCFResult& r);
};

} // namespace mshqc

#endif // MSHQC_SCF_H