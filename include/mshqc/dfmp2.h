#ifndef MSHQC_DFMP2_H
#define MSHQC_DFMP2_H

#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>

/**
 * @file dfmp2.h
 * @brief Density-Fitted MP2 for ROHF
 * 
 * REFERENCES:
 * Feyereisen et al. (1993), Chem. Phys. Lett. 208, 359 - Eq. (7)
 * Weigend et al. (1998), Chem. Phys. Lett. 294, 143 - Eq. (3)
 * 
 * Theory:
 * Approximate (μν|λσ) ≈ Σ_PQ B_μνP [J^-1]_PQ B_λσQ
 * where B_μνP = (μν|P) are 3-center integrals
 * and J_PQ = (P|Q) is auxiliary basis metric
 * 
 * MP2 energy:
 * E = Σ_{ijab} t_ijab <ia|jb>_DF
 * where <ia|jb>_DF uses fitted integrals
 */

namespace mshqc {

/**
 * DF-MP2 result
 */
struct DFMP2Result {
    double e_ss;       // Same-spin correlation
    double e_os;       // Opposite-spin correlation  
    double e_corr;     // Total correlation
    double e_total;    // ROHF + correlation
};

/**
 * Density-Fitted MP2 for ROHF
 * 
 * Uses auxiliary basis (cc-pVTZ-RI) to approximate ERIs
 * Much faster than conventional MP2, same accuracy
 * 
 * REFERENCE: Feyereisen et al. (1993), Eq. (7)
 */
class DFMP2 {
public:
    /**
     * Constructor
     * @param rohf_result ROHF result
     * @param basis Primary basis
     * @param aux_basis Auxiliary basis (e.g. cc-pVTZ-RI)
     * @param integrals Integral engine
     */
    DFMP2(const SCFResult& rohf_result,
          const BasisSet& basis,
          const BasisSet& aux_basis,
          std::shared_ptr<IntegralEngine> integrals);
    
    /**
     * Compute DF-MP2 energy
     */
    DFMP2Result compute();
    
private:
    const SCFResult& rohf_;
    const BasisSet& basis_;
    const BasisSet& aux_basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    
    // Dimensions
    int nbf_;      // primary basis
    int naux_;     // auxiliary basis
    int nocc_;     // occupied orbitals
    int nvir_;     // virtual orbitals
    
    // DF tensors
    Eigen::MatrixXd B_ia_;  // 3-center (ia|P) in MO basis
    Eigen::MatrixXd J_;     // Metric (P|Q)
    Eigen::MatrixXd J_inv_; // Inverse metric [J^-1/2]_PQ
    
    /**
     * Build auxiliary metric and invert
     * J_PQ = (P|Q)
     * 
     * REFERENCE: Feyereisen et al. (1993), Eq. (8)
     */
    void compute_metric();
    
    /**
     * Transform 3-center to MO basis
     * B^P_ia = Σ_μν C_μi (μν|P) C_νa
     */
    void transform_3center();
    
    /**
     * Compute DF-MP2 energy components
     * 
     * REFERENCE: Feyereisen et al. (1993), Eq. (11)
     * E = Σ t_ijab <ia|jb>_DF
     * where <ia|jb>_DF = Σ_PQ B^P_ia [J^-1]_PQ B^Q_jb
     */
    double compute_ss_energy();
    double compute_os_energy();
};

} // namespace mshqc

#endif // MSHQC_DFMP2_H
