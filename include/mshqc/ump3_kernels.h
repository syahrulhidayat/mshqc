/**
 * @file include/mshqc/ump3_kernels.h
 * @brief Optimized MP3 Contraction Kernels (BLAS/Eigen Backend)
 */

#ifndef MSHQC_UMP3_KERNELS_H
#define MSHQC_UMP3_KERNELS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace kernels {

    // --- LADDER TERM KERNELS ---
    
    // Menghitung E = factor * Sum(T_ijab * W_ijab)
    // Dimana W = V * T (Matrix Multiplication)
    // Input Tensor: (Occ, Occ, Vir, Vir)
    double contract_ladder_pp(const Eigen::Tensor<double, 4>& T2, 
                              const Eigen::Tensor<double, 4>& V_vvvv, 
                              double factor);

    double contract_ladder_hh(const Eigen::Tensor<double, 4>& T2, 
                              const Eigen::Tensor<double, 4>& V_oooo, 
                              double factor);

    // --- RING TERM KERNELS ---
    
    // Menghitung E = factor * Sum(T_iajb * W_iajb)
    // Membutuhkan permutasi tensor T2: (i,j,a,b) -> (i,a,j,b) secara efisien
    double contract_ring_ph(const Eigen::Tensor<double, 4>& T2, 
                            const Eigen::Tensor<double, 4>& V_ovov, 
                            double factor);

    // Kernel Spesial untuk Mixed Spin (AB) Ring Term 3 & 4
    double contract_ring_mixed_exchange(const Eigen::Tensor<double, 4>& T2_AA,
                                        const Eigen::Tensor<double, 4>& T2_BB,
                                        const Eigen::Tensor<double, 4>& V_AB_1, // (k,c,j,b)
                                        const Eigen::Tensor<double, 4>& V_AB_2, // (i,k,a,c)
                                        const Eigen::Tensor<double, 4>& T2_AB,
                                        double factor);

} // namespace kernels
} // namespace mshqc

#endif // MSHQC_UMP3_KERNELS_H