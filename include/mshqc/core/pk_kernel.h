#ifndef MSHQC_CORE_PK_KERNEL_H
#define MSHQC_CORE_PK_KERNEL_H

#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {

class PKKernel {
public:

    static void contract_JK_generic(const double* __restrict__ integrals,
                                    const double* __restrict__ P_a, const double* __restrict__ P_b,
                                    double* __restrict__ F_a, double* __restrict__ F_b,
                                    int n1, int n2, int n3, int n4);

    template<int N1, int N2, int N3, int N4>
    static void contract_JK_opt(const double* __restrict__ integrals,
                                const double* __restrict__ P_a, const double* __restrict__ P_b,
                                double* __restrict__ F_a, double* __restrict__ F_b);

    static void contract_K_generic(const double* __restrict__ I,
                                   const double* __restrict__ P,
                                   double* __restrict__ F,
                                   int n1, int n2, int n3, int n4,
                                   double factor);

    template<int N1, int N2, int N3, int N4>
    static void contract_K_opt(const double* __restrict__ I,
                               const double* __restrict__ P,
                               double* __restrict__ F,
                               double factor);

};

}

#endif
