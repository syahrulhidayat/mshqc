#ifndef MSHQC_CORE_BLOCK_FOCK_H
#define MSHQC_CORE_BLOCK_FOCK_H

#include <Eigen/Dense>
#include <vector>
#include "mshqc/ints/block_data.h"
#ifdef I
#undef I
#endif

namespace mshqc {

class BlockFockBuilder {
public:
    /**
     * @brief Menghitung kontribusi Coulomb (J) menggunakan BLAS Level 3.
     * Operasi: F_block += I_batch * P_block
     * * @param I_batch Pointer ke buffer integral (Flattened Matrix: dim_row x dim_col)
     * @param P_block Pointer ke buffer density (Flattened Vector: dim_col x 1)
     * @param F_block Pointer ke buffer Fock output (Flattened Vector: dim_row x 1)
     * @param dim_row Dimensi baris (Size Block A * Size Block B)
     * @param dim_col Dimensi kolom (Size Block C * Size Block D)
     */
    static void accumulate_J_blas(const double* I_batch, 
                                  const double* P_block, 
                                  double* F_block,
                                  int dim_row, int dim_col);

    /**
     * @brief Mengambil sub-blok dari Matrix Density global dan menyimpannya di buffer kontigu.
     * Penting agar memori akses saat DGEMM linear.
     */
    static void pack_density_block(const Eigen::MatrixXd& P, 
                                   const ShellBlock& C, const ShellBlock& D, 
                                   double* buffer);

    /**
     * @brief Menyebarkan hasil buffer Fock lokal ke Matrix Fock global (Thread-Safe via Atomic).
     */
    static void scatter_fock_block_atomic(Eigen::MatrixXd& F, 
                                          const ShellBlock& A, const ShellBlock& B, 
                                          const double* buffer);
};

} // namespace mshqc

#endif