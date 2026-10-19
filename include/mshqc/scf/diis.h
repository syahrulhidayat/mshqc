// Copyright 2026 Muhamad Syahrul Hidayat
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
