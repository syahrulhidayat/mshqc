#ifndef MSHQC_INTS_MATH_UTILS_H
#define MSHQC_INTS_MATH_UTILS_H

// 1. Core Eigen (Ringan)
#include <Eigen/Dense>

// 2. Tensor (Harus di header karena Template Class)
//    Kita tidak bisa menyembunyikan ini di .cc jika tipe data Tensor
//    digunakan di struct/class file lain.
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace utils {

    // --- TYPEDEFS (Opsional, biar koding lebih singkat) ---
    using Tensor4D = Eigen::Tensor<double, 4>;
    using Tensor2D = Eigen::Tensor<double, 2>;

    // --- FUNGSI BERAT (Deklarasi Saja) ---
    // Implementasinya ada di .cc sehingga file lain tidak perlu
    // meng-include <unsupported/Eigen/MatrixFunctions>
    Eigen::MatrixXd matrix_exponential(const Eigen::MatrixXd& mat);

    // --- HELPER TENSOR (Opsional) ---
    // Contoh fungsi helper untuk mereset tensor dengan aman
    void set_zero(Tensor4D& tensor);

} // namespace utils
} // namespace mshqc

#endif // MSHQC_INTS_MATH_UTILS_H