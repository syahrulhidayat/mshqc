 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //     http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 // ==============================================================================

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
