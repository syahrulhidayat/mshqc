#ifndef MSHQC_DFMP2_H
#define MSHQC_DFMP2_H

#include "mshqc/scf/scf.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {

struct DFMP2Result {
    double e_ss;

    double e_os;

    double e_corr;

    double e_total;

};

class DFMP2 {
public:

    DFMP2(const SCFResult& rohf_result,
          const BasisSet& basis,
          const BasisSet& aux_basis,
          std::shared_ptr<IntegralEngine> integrals);

    DFMP2Result compute();

private:
    const SCFResult& rohf_;
    const BasisSet& basis_;
    const BasisSet& aux_basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    int nbf_;

    int naux_;

    int nocc_;

    int nvir_;

    Eigen::MatrixXd B_ia_;

    Eigen::MatrixXd J_;

    Eigen::MatrixXd J_inv_;

    void compute_metric();

    void transform_3center();

    double compute_ss_energy();
    double compute_os_energy();
};

}

#endif
