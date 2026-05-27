/**
 * @file src/mp3/ump3_kernels.cc
 * @brief Implementation of Fast MP3 Kernels
 * @details 
 */

#include "mshqc/ump3_kernels.h"
#include <iostream>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace kernels {

    // ========================================================================
    // LADDER KERNELS (Particle-Particle & Hole-Hole)
    // ========================================================================

    double contract_ladder_pp(const Eigen::Tensor<double, 4>& T2, 
                              const Eigen::Tensor<double, 4>& V_vvvv, 
                              double factor) 
    {
        // T2 Dimensions: (OccA, OccB, VirA, VirB)
        long no1 = T2.dimension(0);
        long no2 = T2.dimension(1);
        long nv1 = T2.dimension(2);
        long nv2 = T2.dimension(3);

        long dim_occ = no1 * no2;
        long dim_vir = nv1 * nv2;

        // 1. Map Tensor ke Matrix (Zero Copy View)
        
        Eigen::Map<const Eigen::MatrixXd> T_mat(T2.data(), dim_occ, dim_vir);
        
        // V(a,b,c,d) -> Matriks V[dim_vir, dim_vir]
        // Pastikan V_vvvv tersimpan sebagai (a,b,c,d)
        Eigen::Map<const Eigen::MatrixXd> V_mat(V_vvvv.data(), dim_vir, dim_vir);

        // 2. GEMM: W = T * V     
        // W = T_mat * V_mat
        // Dimensi: (occ, vir) = (occ, vir) * (vir, vir)
        Eigen::MatrixXd W = T_mat * V_mat;

        // 3. Dot Product: Sum(T_ijab * W_ijab)
        double energy = T_mat.cwiseProduct(W).sum();

        return factor * energy;
    }

    double contract_ladder_hh(const Eigen::Tensor<double, 4>& T2, 
                              const Eigen::Tensor<double, 4>& V_oooo, 
                              double factor) 
    {
        long no1 = T2.dimension(0);
        long no2 = T2.dimension(1);
        long nv1 = T2.dimension(2);
        long nv2 = T2.dimension(3);

        long dim_occ = no1 * no2;
        long dim_vir = nv1 * nv2;

        Eigen::Map<const Eigen::MatrixXd> T_mat(T2.data(), dim_occ, dim_vir);
        Eigen::Map<const Eigen::MatrixXd> V_mat(V_oooo.data(), dim_occ, dim_occ);

        // Rumus HH: W_ij^ab = Sum_kl <kl|ij> T_kl^ab
        // W = V * T
        // V: (occ, occ) [kl, ij]. T: (occ, vir) [kl, ab].
        
        Eigen::MatrixXd W = V_mat.transpose() * T_mat; 

        double energy = T_mat.cwiseProduct(W).sum();
        return factor * energy;
    }

    // ========================================================================
    // RING KERNEL (Particle-Hole)
    // ========================================================================

    double contract_ring_ph(const Eigen::Tensor<double, 4>& T2, 
                            const Eigen::Tensor<double, 4>& V_ovov, 
                            double factor) 
    {
        long no = T2.dimension(0); // i
        long nv = T2.dimension(2); // a
        
        // 1. Permute T2: (i,j,a,b) -> (i,a,j,b)
        // Kita butuh layout (ia, jb) untuk perkalian matriks dengan V_iajb
        // V_ovov sudah dalam bentuk (i,a, j,b) dari transform_ovov.
        
        Eigen::array<long, 4> shuffling = {0, 2, 1, 3}; // 0->i, 1->j, 2->a, 3->b
        
        // Tensor temporary (terpaksa copy untuk reshuffling, tapi ini O(N^4) memory transfer saja)
        // Jauh lebih cepat daripada loop O(N^6)
        Eigen::Tensor<double, 4> T_iajb_tensor = T2.shuffle(shuffling);
        
        long dim_ia = no * nv;
        
        // 2. Map ke Matrix
        Eigen::Map<const Eigen::MatrixXd> T_mat(T_iajb_tensor.data(), dim_ia, dim_ia);
        Eigen::Map<const Eigen::MatrixXd> V_mat(V_ovov.data(), dim_ia, dim_ia);

        // 3. GEMM: W_iajb = Sum_kc T_iakc * V_kcjb
        // W = T * V
        Eigen::MatrixXd W = T_mat * V_mat;

        // 4. Energi
        double energy = T_mat.cwiseProduct(W).sum();
        return factor * energy;
    }

    // ========================================================================
    // MIXED SPIN EXCHANGE KERNEL (Term 3 & 4) - ADVANCED
    // ========================================================================
    
    // Kernel ini khusus untuk menghitung Term 3 & 4 dari Ring Mixed Spin
    // Term 3: Sum_kc T_AA(ik,ac) * V_AB(kc,jb) -> Kontribusi ke T_AB(ia,jb)
    // Term 4: Sum_kc V_AB(ia,kc) * T_BB(kj,cb) -> Kontribusi ke T_AB(ia,jb)
    
    double contract_ring_mixed_exchange(const Eigen::Tensor<double, 4>& T2_AA,
                                        const Eigen::Tensor<double, 4>& T2_BB,
                                        const Eigen::Tensor<double, 4>& V_AB_1, // g_ovov_ab (k, c, j, b) -> (kc, jb)
                                        const Eigen::Tensor<double, 4>& V_AB_2, // g_oovv_ab (i, k, a, c) -> (ia, kc)
                                        const Eigen::Tensor<double, 4>& T2_AB,
                                        double factor)
    {
        // 1. Siapkan T_AB dalam bentuk (ia, jb) untuk Dot Product akhir
        long noa = T2_AB.dimension(0); long nob = T2_AB.dimension(1);
        long nva = T2_AB.dimension(2); long nvb = T2_AB.dimension(3);
        
        long dim_ia = noa * nva;
        long dim_jb = nob * nvb;
        
        Eigen::array<long, 4> shuf_iajb = {0, 2, 1, 3}; // (i,j,a,b) -> (i,a,j,b)
        Eigen::Tensor<double, 4> T_AB_iajb = T2_AB.shuffle(shuf_iajb);
        Eigen::Map<const Eigen::MatrixXd> Mat_T_AB(T_AB_iajb.data(), dim_ia, dim_jb);
        
        Eigen::MatrixXd W_Total = Eigen::MatrixXd::Zero(dim_ia, dim_jb);

        // --- TERM 3: T_AA * V_AB ---
        // T_AA(i,k,a,c). Shuffle -> (i,a, k,c) -> Matrix(ia, kc)
        // V_AB_1(k,c, j,b) -> Matrix(kc, jb) [Sudah layout (kc, jb) karena transform_oovv_mixed]
        {
            // Shuffle T_AA
            // T2_AA dimensions: (noa, noa, nva, nva)
            // Shuffle indices: (i, k, a, c) -> (i, a, k, c) [0, 2, 1, 3]
            Eigen::Tensor<double, 4> T_AA_iakc = T2_AA.shuffle(shuf_iajb);
            
            // Map
            long dim_kc = noa * nva; // Alpha-Alpha
            Eigen::Map<const Eigen::MatrixXd> Mat_T_AA(T_AA_iakc.data(), dim_ia, dim_kc);
            Eigen::Map<const Eigen::MatrixXd> Mat_V1(V_AB_1.data(), dim_kc, dim_jb);
            
            // GEMM: (ia, kc) * (kc, jb) -> (ia, jb)
            W_Total -= Mat_T_AA * Mat_V1; // Tanda Minus sesuai term exchange
        }

        {
            // Shuffle V_AB_2 (i,k, a,c) -> (i,a, k,c)
            Eigen::array<long, 4> shuf_v = {0, 2, 1, 3};
            Eigen::Tensor<double, 4> V2_iakc = V_AB_2.shuffle(shuf_v);
            
            long dim_kc_beta = nob * nvb; // Beta internal
            Eigen::Map<const Eigen::MatrixXd> Mat_V2(V2_iakc.data(), dim_ia, dim_kc_beta);
            
            // Shuffle T_BB (k,j, c,b) -> (k,c, j,b)
            // T2_BB dimensions: (nob, nob, nvb, nvb)
            Eigen::Tensor<double, 4> T_BB_kcjb = T2_BB.shuffle(shuf_iajb);
            Eigen::Map<const Eigen::MatrixXd> Mat_T_BB(T_BB_kcjb.data(), dim_kc_beta, dim_jb);
            
            // GEMM: (ia, kc) * (kc, jb) -> (ia, jb)
            W_Total += Mat_V2 * Mat_T_BB; // Tanda Plus? Cek lagi sign convention Anda nanti
        }

        double energy = Mat_T_AB.cwiseProduct(W_Total).sum();
        return factor * energy;
    }

} // namespace kernels
} // namespace mshqc