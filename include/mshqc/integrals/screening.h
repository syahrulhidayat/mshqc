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

#ifndef MSHQC_INTEGRALS_SCREENING_H
#define MSHQC_INTEGRALS_SCREENING_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor> 
#include <vector>
#include <memory>
#include <cmath>
#include "mshqc/basis.h" 
#ifdef I
#undef I
#endif

namespace mshqc {

// [PENTING] Forward Declaration: Memberi tahu compiler bahwa class ini ada
class IntegralEngine; 

namespace integrals {

struct ShellPair {
    int sh_a;
    int sh_b;
    double max_val; 
    Eigen::Vector3d center; 
};

class Screening {
public:
    Screening(const BasisSet& basis);

    void reset();
    void print_stats(double threshold) const;

    // [BARU] Tambahkan deklarasi ini agar sesuai dengan screening.cc
    void compute(std::shared_ptr<mshqc::IntegralEngine> integrals);

    // Fungsi lama (opsional, bisa dihapus jika tidak dipakai)
    void build_from_eri_tensor(const Eigen::Tensor<double, 4>& eri);

    std::vector<ShellPair> get_significant_pairs(double threshold) const;

    // Screening Functions
    bool is_significant(int sh_a, int sh_b, int sh_c, int sh_d, double threshold) const;
    bool is_significant(const ShellPair& P, const ShellPair& Q, double threshold) const;
    double get_schwarz_val(int sh_a, int sh_b) const;
    double max_schwarz() const { return max_Q_; }

private:
    const BasisSet& basis_;
    int nshells_;
    
    Eigen::MatrixXd Q_matrix_; 
    double max_Q_;
    std::vector<Eigen::Vector3d> shell_centers_;

    void init_shell_centers();
};

} // namespace integrals
} // namespace mshqc

#endif // MSHQC_INTEGRALS_SCREENING_