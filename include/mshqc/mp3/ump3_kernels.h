#ifndef MSHQC_UMP3_KERNELS_H
#define MSHQC_UMP3_KERNELS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace kernels {

    double contract_ladder_pp(const Eigen::Tensor<double, 4>& T2,
                              const Eigen::Tensor<double, 4>& V_vvvv,
                              double factor);

    double contract_ladder_hh(const Eigen::Tensor<double, 4>& T2,
                              const Eigen::Tensor<double, 4>& V_oooo,
                              double factor);

    double contract_ring_ph(const Eigen::Tensor<double, 4>& T2,
                            const Eigen::Tensor<double, 4>& V_ovov,
                            double factor);

    double contract_ring_mixed_exchange(const Eigen::Tensor<double, 4>& T2_AA,
                                        const Eigen::Tensor<double, 4>& T2_BB,
                                        const Eigen::Tensor<double, 4>& V_AB_1,

                                        const Eigen::Tensor<double, 4>& V_AB_2,

                                        const Eigen::Tensor<double, 4>& T2_AB,
                                        double factor);

}

}

#endif
