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

    static void accumulate_J_blas(const double* I_batch,
                                  const double* P_block,
                                  double* F_block,
                                  int dim_row, int dim_col);

    static void pack_density_block(const Eigen::MatrixXd& P,
                                   const ShellBlock& C, const ShellBlock& D,
                                   double* buffer);

    static void scatter_fock_block_atomic(Eigen::MatrixXd& F,
                                          const ShellBlock& A, const ShellBlock& B,
                                          const double* buffer);
};

}

#endif
