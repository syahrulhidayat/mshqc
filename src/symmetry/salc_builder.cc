#include "mshqc/symmetry/salc_builder.h"
#include <random>
#include <iostream>
#include <algorithm>

namespace mshqc {

// Ubah konstruktor di sini
SalcBuilder::SalcBuilder(BasisSymmetrizer* sym) : sym_(sym) {}

std::pair<Eigen::MatrixXd, std::vector<int>> SalcBuilder::build_salc(const Eigen::MatrixXd& S) {
    int nbf = S.rows();
    
    std::mt19937 gen(42); 
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbf, nbf);
    for (int i = 0; i < nbf; ++i) {
        for (int j = 0; j <= i; ++j) {
            double val = dist(gen);
            M(i, j) = val;
            M(j, i) = val;
        }
    }
    
    sym_->symmetrize(M);
    
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(M, S);
    Eigen::MatrixXd C_salc = es.eigenvectors();
    
    std::vector<int> irreps = sym_->assign_mo_irreps(C_salc, 1e-5);
    
    std::vector<std::pair<int, int>> sort_data;
    for (int i = 0; i < nbf; ++i) {
        sort_data.push_back({irreps[i], i});
    }
    
    std::stable_sort(sort_data.begin(), sort_data.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first < b.first;
        });
        
    Eigen::MatrixXd X_salc = Eigen::MatrixXd::Zero(nbf, nbf);
    std::vector<int> sorted_irreps(nbf);
    
    for (int i = 0; i < nbf; ++i) {
        X_salc.col(i) = C_salc.col(sort_data[i].second);
        sorted_irreps[i] = sort_data[i].first;
    }
    
    for (int i = 0; i < nbf; ++i) {
        double norm = std::sqrt(X_salc.col(i).transpose() * S * X_salc.col(i));
        X_salc.col(i) /= norm;
    }

    return {X_salc, sorted_irreps};
}

} // namespace mshqc