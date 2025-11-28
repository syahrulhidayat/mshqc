/**
 * @file mp2.h
 * @brief Restricted Open-shell Møller-Plesset 2nd order perturbation theory
 * 
 * THEORY BACKGROUND:
 * MP2 treats electron correlation through 2nd-order perturbation theory,
 * calculating energy correction from double excitations.
 * 
 * For open-shell systems (ROHF reference), we have three spin cases:
 * - Same-spin alpha-alpha (αα)
 * - Mixed-spin alpha-beta (αβ)  
 * - Same-spin beta-beta (ββ)
 * 
 * REFERENCES:
 * [1] Møller, C. & Plesset, M. S., Phys. Rev. 46, 618 (1934)
 *     - Original MP perturbation theory
 * 
 * [2] Szabo, A. & Ostlund, N. S., Modern Quantum Chemistry (1996)
 *     Dover Publications, ISBN: 0-486-69186-1
 *     - Chapter 6: Many-Body Perturbation Theory
 *     - Section 6.4: Møller-Plesset Perturbation Theory
 * 
 * [3] Bozkaya, U., Turney, J. M., Yamaguchi, Y., Schaefer III, H. F., 
 *     & Sherrill, C. D., J. Chem. Phys. 135, 104103 (2011)
 *     DOI: 10.1063/1.3631129
 *     - Section II.A: ROHF-based MP2
 *     - Equations for spin-adapted amplitudes
 * 
 * [4] Pople, J. A., Binkley, J. S., & Seeger, R., 
 *     Int. J. Quantum Chem. Symp. 10, 1 (1976)
 *     - Spin-restricted open-shell MP2
 */

#ifndef MSHQC_MP2_H
#define MSHQC_MP2_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

/**
 * @brief MP2 calculation results
 */
struct MP2Result {
    double energy_scf;          ///< Reference SCF energy (Ha)
    double energy_mp2_ss;       ///< Same-spin correlation (Ha)
    double energy_mp2_os;       ///< Opposite-spin correlation (Ha)
    double energy_mp2_corr;     ///< Total MP2 correlation (Ha)
    double energy_total;        ///< Total MP2 energy (Ha)
    
    int n_occ_alpha;           ///< Number of occupied α orbitals
    int n_occ_beta;            ///< Number of occupied β orbitals
    int n_virt_alpha;          ///< Number of virtual α orbitals
    int n_virt_beta;           ///< Number of virtual β orbitals
};

/**
 * @brief Restricted Open-shell MP2 implementation
 * 
 * THEORY:
 * MP2 correlation energy is calculated from double excitation amplitudes:
 * 
 * E_corr^MP2 = E_SS + E_OS
 * 
 * where:
 * E_SS  = same-spin contribution (αα + ββ)
 * E_OS  = opposite-spin contribution (αβ)
 * 
 * AMPLITUDE EQUATIONS:
 * The MP2 amplitude for excitation from occupied i,j to virtual a,b is:
 * 
 * t_ijab = <ij||ab> / (ε_i + ε_j - ε_a - ε_b)
 * 
 * REFERENCE: Szabo & Ostlund (1996), Eq. (6.63), p. 352
 * 
 * For ROHF, we use spin-orbital notation with three cases:
 * 
 * 1. Alpha-alpha (same-spin):
 *    t^αα_ijab = <ij||ab>^αα / D^αα_ijab
 *    where <ij||ab> = <ij|ab> - <ij|ba> (antisymmetrized)
 * 
 * 2. Alpha-beta (opposite-spin):
 *    t^αβ_iJaB = <iJ|aB>^αβ / D^αβ_iJaB
 *    where J,B are β spin indices (no antisymmetrization)
 * 
 * 3. Beta-beta (same-spin):
 *    t^ββ_ijab = <ij||ab>^ββ / D^ββ_ijab
 * 
 * REFERENCE: Bozkaya et al. (2011), Eq. (2-4)
 * 
 * ENERGY EXPRESSIONS:
 * 
 * E^αα_corr = (1/4) Σ_ijab t^αα_ijab × <ij||ab>^αα
 * E^αβ_corr = Σ_iJaB t^αβ_iJaB × <iJ|aB>^αβ
 * E^ββ_corr = (1/4) Σ_ijab t^ββ_ijab × <ij||ab>^ββ
 * 
 * E_MP2 = E^αα_corr + E^αβ_corr + E^ββ_corr
 * 
 * REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
 * REFERENCE: Bozkaya et al. (2011), Eq. (5)
 * 
 * The factor 1/4 accounts for double counting in same-spin cases.
 * 
 * ORBITAL ENERGY DENOMINATORS:
 * 
 * D^αα_ijab = ε^α_i + ε^α_j - ε^α_a - ε^α_b
 * D^αβ_iJaB = ε^α_i + ε^β_J - ε^α_a - ε^β_B
 * D^ββ_ijab = ε^β_i + ε^β_j - ε^β_a - ε^β_b
 * 
 * REFERENCE: Szabo & Ostlund (1996), Eq. (6.64), p. 352
 */
class ROMP2 {
public:
    /**
     * @brief Construct ROMP2 calculator from converged SCF
     * @param scf_result Converged SCF calculation result
     * @param integrals Integral engine (for AO integrals)
     * 
     * Note: SCF must be converged before calling MP2
     */
    ROMP2(const SCFResult& scf_result, 
          std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * @brief Run MP2 correlation calculation
     * @return MP2Result containing energies and amplitudes info
     * 
     * ALGORITHM:
     * 1. Transform AO integrals to MO basis
     * 2. Compute orbital energy denominators
     * 3. Calculate t2 amplitudes for each spin case
     * 4. Compute correlation energy contributions
     * 5. Sum to get total MP2 energy
     * 
     * REFERENCE: Szabo & Ostlund (1996), Section 6.4, pp. 350-359
     */
    MP2Result compute();
    
private:
    // SCF reference data
    SCFResult scf_result_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    size_t nbasis_;
    int n_occ_alpha_;
    int n_occ_beta_;
    int n_virt_alpha_;
    int n_virt_beta_;
    
    // MO integrals (transformed from AO)
    Eigen::Tensor<double, 4> eri_mo_aaaa_;  // <ij|ab>^αα in MO basis
    Eigen::Tensor<double, 4> eri_mo_aabb_;  // <iJ|aB>^αβ in MO basis
    Eigen::Tensor<double, 4> eri_mo_bbbb_;  // <ij|ab>^ββ in MO basis
    
    /**
     * @brief Transform AO integrals to MO basis
     * 
     * MO integral: <pq|rs>_MO = Σ_μνλσ C_μp C_νq <μν|λσ>_AO C_λr C_σs
     * 
     * This is a four-index transformation (N^8 scaling naively).
     * We use intermediate transformations to reduce to N^5:
     * 
     * Step 1: (μν|λσ) → (pν|λσ)  (transform first index)
     * Step 2: (pν|λσ) → (pq|λσ)  (transform second index)
     * Step 3: (pq|λσ) → (pq|rσ)  (transform third index)
     * Step 4: (pq|rσ) → (pq|rs)  (transform fourth index)
     * 
     * REFERENCE: Helgaker et al. (2000), Molecular Electronic-Structure Theory
     *            Section 9.6.2: Four-index transformation
     * 
     * REFERENCE: Szabo & Ostlund (1996), Problem 6.9, p. 360
     */
    void transform_integrals();
    
    /**
     * @brief Compute same-spin MP2 correlation (αα case)
     * 
     * EQUATION:
     * E^αα = (1/4) Σ_{i<j}^{n_α} Σ_{a<b}^{n_virt_α} 
     *        [<ij|ab> - <ij|ba>]² / (ε_i + ε_j - ε_a - ε_b)
     * 
     * Note: The antisymmetrized integral <ij||ab> = <ij|ab> - <ij|ba>
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
     * REFERENCE: Bozkaya et al. (2011), Eq. (5) - same-spin term
     * 
     * @return Same-spin alpha correlation energy (Ha)
     */
    double compute_mp2_ss_alpha();
    
    /**
     * @brief Compute same-spin MP2 correlation (ββ case)
     * 
     * EQUATION:
     * E^ββ = (1/4) Σ_{i<j}^{n_β} Σ_{a<b}^{n_virt_β}
     *        [<ij|ab> - <ij|ba>]² / (ε_i + ε_j - ε_a - ε_b)
     * 
     * REFERENCE: Same as compute_mp2_ss_alpha(), but for β spin
     * 
     * @return Same-spin beta correlation energy (Ha)
     */
    double compute_mp2_ss_beta();
    
    /**
     * @brief Compute opposite-spin MP2 correlation (αβ case)
     * 
     * EQUATION:
     * E^αβ = Σ_i^{n_α} Σ_J^{n_β} Σ_a^{n_virt_α} Σ_B^{n_virt_β}
     *        <iJ|aB>² / (ε^α_i + ε^β_J - ε^α_a - ε^β_B)
     * 
     * Note: No antisymmetrization for opposite-spin case
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (6.74), p. 354
     * REFERENCE: Bozkaya et al. (2011), Eq. (5) - opposite-spin term
     * 
     * @return Opposite-spin correlation energy (Ha)
     */
    double compute_mp2_os();
    
    /**
     * @brief Get antisymmetrized integral <ij||ab> = <ij|ab> - <ij|ba>
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (2.34), p. 70
     *            Antisymmetrized two-electron integral
     */
    double get_antisym_integral(const Eigen::Tensor<double, 4>& eri,
                                int i, int j, int a, int b);
};

/**
 * @brief ROHF-based MBPT(2) with semi-canonical orbitals
 * 
 * THEORY:
 * Semi-canonical MP2 uses block-diagonalized Fock matrix and includes
 * singles (T1) correction. Matches Psi4's "ROHF-MP2" implementation.
 * 
 * Key differences from canonical ROMP2:
 * - Semi-canonical orbitals (Fock block-diagonal)
 * - T1 amplitudes (singles correction)
 * - Typically smaller correlation energy
 * 
 * REFERENCES:
 * [1] Knowles, P. J., Hampel, C., & Werner, H.-J.,
 *     Chem. Phys. Lett. 186, 130 (1991)
 *     - Eq. (3): Semi-canonical transformation
 *     - Eq. (7): T1 amplitudes: t_i^a = F_ia / (ε_i - ε_a)
 *     - Eq. (8): T1 energy: E_T1 = Σ_ia t_ia F_ai
 *     - Eq. (9)-(10): Modified T2 with semi-canonical denominators
 * 
 * [2] Szabo & Ostlund (1996), Section 3.8.7
 */
class ROHF_MBPT2 {
public:
    ROHF_MBPT2(const SCFResult& scf_result,
               std::shared_ptr<IntegralEngine> integrals);
    
    MP2Result compute();
    
private:
    SCFResult scf_result_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    size_t nbasis_;
    int n_c_;   // closed (doubly occ)
    int n_o_;   // open (singly occ)
    int n_v_;   // virtual
    
    // Semi-canonical orbitals
    Eigen::MatrixXd C_semi_;
    Eigen::VectorXd eps_semi_;
    
    // T1 amplitudes
    Eigen::MatrixXd t1_;  // (n_occ, n_virt)
    
    // REFERENCE: Knowles et al. (1991), Eq. (3)
    // Block-diagonalize Fock: rotate orbitals within closed/open/virtual
    void semi_canonical_transform();
    
    // REFERENCE: Knowles et al. (1991), Eq. (7)
    // t_i^a = F_ia / (ε_i - ε_a)
    void compute_t1_amplitudes();
    
    // REFERENCE: Knowles et al. (1991), Eq. (8)
    // E_T1 = Σ_ia t_ia F_ai
    double t1_energy();
    
    // REFERENCE: Knowles et al. (1991), Eq. (9)-(10)
    // MP2 correlation with semi-canonical denominators
    double compute_mp2_correlation();
};

/**
 * @brief Orbital-Optimized MP2 (OMP2)
 * 
 * THEORY:
 * OMP2 optimizes both orbitals and amplitudes simultaneously,
 * improving upon standard MP2 by relaxing orbital constraint.
 * 
 * Iterative procedure:
 * 1. Compute MP2 energy with current orbitals
 * 2. Calculate orbital gradient ∂L/∂κ
 * 3. Rotate orbitals via exp(-κ)
 * 4. Repeat until convergence
 * 
 * REFERENCES:
 * [1] Bozkaya, U. & Sherrill, C. D.,
 *     J. Chem. Phys. 139, 054104 (2013)
 *     - Eq. (11): Orbital gradient
 *     - Eq. (15): Lagrangian L = E_SCF + E_MP2
 *     - Eq. (20)-(22): Orbital update equations
 * 
 * [2] Lochan, R. C., Shao, Y., & Head-Gordon, M.,
 *     J. Chem. Phys. 126, 164101 (2007)
 *     - Original OMP2 formulation
 * 
 * [3] Helgaker, T., Jørgensen, P., & Olsen, J.,
 *     Molecular Electronic-Structure Theory (2000)
 *     - Section 10.8: Orbital rotation parametrization
 */
class OMP2 {
public:
    OMP2(const Molecule& mol,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         const SCFResult& scf_guess);
    
    MP2Result compute();
    
private:
    Molecule mol_;
    BasisSet basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_;  // Current orbitals
    
    int max_iter_ = 50;
    double conv_thresh_ = 1e-6;
    
    // Dimensions
    size_t nbf_;  // basis functions
    int na_, nb_;  // occ alpha, beta
    int va_, vb_;  // virt alpha, beta
    
    // MO integrals (transformed)
    Eigen::Tensor<double, 4> g_aa_, g_ab_, g_bb_;  // <ij|ab> in MO
    Eigen::Tensor<double, 4> g_mo_full_;  // Full MO basis <pq|rs>
    
    // MP2 amplitudes
    Eigen::Tensor<double, 4> t2_aa_, t2_ab_, t2_bb_;  // t_ijab
    
    // REFERENCE: Helgaker et al. (2000), Section 9.6.2
    // Transform AO → MO: <pq|rs> = Σ C_μp C_νq <μν|λσ> C_λr C_σs
    void xform_ints();
    void xform_full_mo();  // Full MO integrals for Fock
    
    // REFERENCE: Bozkaya et al. (2011), Eq. (2-5)
    // MP2 correlation with current orbitals
    double mp2_energy(double& e_ss, double& e_os);
    
    // REFERENCE: Bozkaya & Sherrill (2013), Eq. (13-14)
    // Build 1-RDM from MP2 amplitudes: γ = γ_HF + γ_MP2
    Eigen::MatrixXd build_opdm();
    
    // REFERENCE: Helgaker et al. (2000), Section 10.3
    // Generalized Fock: G_pq = h_pq + Σ_rs γ_rs <pr||qs>
    Eigen::MatrixXd build_gfock(const Eigen::MatrixXd& gamma);
    
    // REFERENCE: Bozkaya & Sherrill (2013), Eq. (11)
    // κ_pq = ∂L/∂κ_pq where L = E_SCF + E_MP2
    Eigen::MatrixXd orbital_gradient();
    
    // REFERENCE: Helgaker et al. (2000), Section 10.8
    // C_new = C_old exp(-κ)
    void rotate_orbitals(const Eigen::MatrixXd& kappa);
    
    bool converged(const Eigen::MatrixXd& kappa);
};

/**
 * @brief Transform ROHF orbitals to semi-canonical basis
 * 
 * THEORY:
 * Semi-canonical transformation diagonalizes Fock matrix within
 * closed-shell, open-shell, and virtual orbital subspaces.
 * This gives proper energy denominators for ROHF-MP2.
 * 
 * REFERENCE: Knowles et al. (1991), Chem. Phys. Lett. 186, 130, Eq. (3)
 * REFERENCE: Pople et al. (1976), Int. J. Quantum Chem. Symp. 10, 1
 * 
 * @param rohf_result Canonical ROHF calculation result
 * @return SCFResult with semi-canonical orbitals and energies
 * 
 * ALGORITHM:
 * 1. Partition Fock into blocks: F = [F_cc F_co F_cv]
 *                                      [F_oc F_oo F_ov]
 *                                      [F_vc F_vo F_vv]
 * 2. Diagonalize F_oo (open-shell)
 * 3. Diagonalize F_vv (virtual) separately for α and β
 * 4. Reconstruct C with transformed orbitals
 */
SCFResult semicanonicalize(const SCFResult& rohf_result);

} // namespace mshqc

#endif // MSHQC_MP2_H
