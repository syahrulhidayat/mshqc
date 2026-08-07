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

    DIIS(int max_vectors = 8);

    void clear();

    void add_iteration(const Eigen::MatrixXd& F,
                       const Eigen::MatrixXd& err,
                       const Eigen::MatrixXd& P);

    Eigen::MatrixXd extrapolate();

private:
    int max_vectors_;

    std::deque<Eigen::MatrixXd> fock_history_;
    std::deque<Eigen::MatrixXd> error_history_;
    std::deque<Eigen::MatrixXd> density_history_;

    Eigen::VectorXd solve_cdiis(const Eigen::MatrixXd& B);
    Eigen::VectorXd solve_ediis();

    Eigen::MatrixXd B_matrix_;

    double compute_ediis_element(int i, int j) const;
    Eigen::VectorXd solve_svd(const Eigen::MatrixXd& A);
};

}

#endif
