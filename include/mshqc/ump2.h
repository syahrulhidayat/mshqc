#ifndef MSHQC_UMP2_H
#define MSHQC_UMP2_H

#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * @file ump2.h
 * @brief Unrestricted MP2 for open-shell systems
 * 
 * REFERENCES:
 * Pople et al. (1976), Int. J. Quantum Chem. 10, 1
 * PySCF ump2.py (github.com/pyscf/pyscf/blob/master/pyscf/mp/ump2.py)
 */

namespace mshqc {

/**
 * UMP2 result structure
 */
struct UMP2Result {
    double e_corr_ss_aa;  // Same-spin αα
    double e_corr_ss_bb;  // Same-spin ββ
    double e_corr_os;     // Opposite-spin αβ
    double e_corr_total;  // Total correlation
    double e_total;       // UHF + correlation
};

/**
 * T2 amplitude tensors for wavefunction analysis
 */
struct T2Amplitudes {
    Eigen::Tensor<double, 4> t2_aa;  // αα amplitudes
    Eigen::Tensor<double, 4> t2_bb;  // ββ amplitudes
    Eigen::Tensor<double, 4> t2_ab;  // αβ amplitudes
};

/**
 * Unrestricted Møller-Plesset 2nd order
 * 
 * REFERENCE: Pople et al. (1976), Int. J. Quantum Chem. 10, 1
 * 
 * Energy components:
 * E_MP2 = E_SS(αα) + E_SS(ββ) + E_OS(αβ)
 * 
 * E_SS(αα) = 0.25 Σ_{ijab} t_{ijab}^α <ij||ab>^α
 * E_SS(ββ) = 0.25 Σ_{ijab} t_{ijab}^β <ij||ab>^β  
 * E_OS(αβ) = Σ_{iJaB} t_{iJaB}^{αβ} <iJ|aB>
 * 
 * Key: α and β have DIFFERENT orbitals and energies!
 */
class UMP2 {
public:
    /**
     * Constructor
     * @param uhf_result UHF SCF result (must contain C_alpha, C_beta, eps_alpha, eps_beta)
     * @param basis Basis set
     * @param integrals Integral engine
     */
    UMP2(const SCFResult& uhf_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * Compute UMP2 energy
     */
    UMP2Result compute();
    
    /**
     * Get T2 amplitudes for wavefunction analysis
     * Must be called after compute()
     */
    T2Amplitudes get_t2_amplitudes() const;
    
private:
    const SCFResult& uhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;       // # basis functions
    int nocc_a_;    // # α occupied
    int nocc_b_;    // # β occupied  
    int nvir_a_;    // # α virtual
    int nvir_b_;    // # β virtual
    
    // MO integrals (ijab notation: i,j=occ, a,b=virt)
    Eigen::Tensor<double, 4> eri_aaaa_;  // <ij|ab>^αα (antisym)
    Eigen::Tensor<double, 4> eri_bbbb_;  // <IJ|AB>^ββ (antisym)
    Eigen::Tensor<double, 4> eri_aabb_;  // <iJ|aB>^αβ (no antisym)
    
    // T2 amplitudes (stored after compute())
    Eigen::Tensor<double, 4> t2_aa_;  // t_ij^ab (αα)
    Eigen::Tensor<double, 4> t2_bb_;  // t_IJ^AB (ββ)
    Eigen::Tensor<double, 4> t2_ab_;  // t_iJ^aB (αβ)
    
    /**
     * Transform AO integrals to MO basis
     * 
     * REFERENCE: Szabo & Ostlund (1996), Eq. (2.282)
     * <pq|rs> = Σ_{μνλσ} C_μp C_νq (μν|λσ) C_λr C_σs
     */
    void transform_integrals();
    
    /**
     * Compute same-spin αα contribution
     * 
     * E_SS(αα) = 0.25 Σ_{i<j,a<b} |<ij||ab>^α|² / D_{ijab}
     * where D = ε_i + ε_j - ε_a - ε_b
     */
    double compute_ss_alpha();
    
    /**
     * Compute same-spin ββ contribution
     */
    double compute_ss_beta();
    
    /**
     * Compute opposite-spin αβ contribution
     * 
     * E_OS = Σ_{i,J,a,B} |<iJ|aB>|² / D_{iJaB}
     */
    double compute_os();
};

} // namespace mshqc

#endif // MSHQC_UMP2_H
