#ifndef MSHQC_SCF_DIIS_H
#define MSHQC_SCF_DIIS_H

#include <Eigen/Dense>
#include <vector>
#include <deque>
#ifdef I
#undef I
#endif

namespace mshqc {

class DIIS {
public:
    // Constructor dengan ukuran history (default 8-10)
    DIIS(int max_vectors = 8);

    // Reset history (panggil saat restart SCF)
    void clear();

    // Tambahkan data iterasi terbaru
    // PENTING: Untuk EDIIS kita butuh Density (P) juga
    void add_iteration(const Eigen::MatrixXd& F, 
                       const Eigen::MatrixXd& err,
                       const Eigen::MatrixXd& P);

    // Menghitung matriks Fock ekstrapolasi
    Eigen::MatrixXd extrapolate();

private:
    int max_vectors_;
    
    // History Storage
    std::deque<Eigen::MatrixXd> fock_history_;
    std::deque<Eigen::MatrixXd> error_history_;
    std::deque<Eigen::MatrixXd> density_history_; // Baru: Untuk EDIIS

    // Solver Helper
    Eigen::VectorXd solve_cdiis(const Eigen::MatrixXd& B);
    Eigen::VectorXd solve_ediis(); // Solver khusus EDIIS
    Eigen::MatrixXd B_matrix_;

    // Utility
    double compute_ediis_element(int i, int j) const;
    Eigen::VectorXd solve_svd(const Eigen::MatrixXd& A);
};

} // namespace mshqc

#endif // MSHQC_SCF_DIIS_H