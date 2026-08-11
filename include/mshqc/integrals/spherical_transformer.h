// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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
// ==============================================================================


#ifndef MSHQC_SPHERICAL_TRANSFORMER_H
#define MSHQC_SPHERICAL_TRANSFORMER_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <map>
#ifdef I
#undef I
#endif

namespace mshqc {

class SphericalTransformer {
public:
    SphericalTransformer();
    ~SphericalTransformer() = default;

    Eigen::MatrixXd transform_1e_matrix(
        const Eigen::MatrixXd& cart_matrix,
        const std::vector<int>& angular_momenta,
        const std::vector<int>& shell_offsets_cart,
        const std::vector<int>& shell_offsets_sph
    ) const;

    std::vector<double> transform_2e_integrals(
        const std::vector<double>& cart_eris,
        const std::vector<int>& angular_momenta,
        int nbf_cart,
        int nbf_sph
    );

    Eigen::MatrixXd transform_mo_coefficients(
        const Eigen::MatrixXd& cart_coeff,
        const std::vector<int>& angular_momenta,
        const std::vector<int>& shell_offsets_cart,
        const std::vector<int>& shell_offsets_sph
    ) const;

    Eigen::MatrixXd get_transformation_matrix(int l)const;

    int get_cartesian_size(int l) const;
    int get_spherical_size(int l) const;

    bool is_spherical_basis(const std::vector<int>& angular_momenta) const;
    int count_spherical_functions(const std::vector<int>& angular_momenta) const;
    int count_cartesian_functions(const std::vector<int>& angular_momenta) const;

    void transform_eri_shell_quartet(
        const double* cart_eri,
        double* sph_eri,
        const Eigen::MatrixXd& T1,
        const Eigen::MatrixXd& T2,
        const Eigen::MatrixXd& T3,
        const Eigen::MatrixXd& T4,
        int n1_cart, int n2_cart, int n3_cart, int n4_cart,
        int n1_sph, int n2_sph, int n3_sph, int n4_sph
    );

private:

    std::map<int, Eigen::MatrixXd> transformation_matrices_;

    void initialize_transformation_matrices();

    Eigen::MatrixXd get_s_transform();

    Eigen::MatrixXd get_p_transform();

    Eigen::MatrixXd get_d_transform();

    Eigen::MatrixXd get_f_transform();

    Eigen::MatrixXd get_g_transform();

    int cartesian_index(int l, int i, int j, int k) const;
    int spherical_index(int l, int m) const;
};

class BasisTransformationHelper {
public:
    struct ShellInfo {
        int angular_momentum;
        int cart_offset;
        int sph_offset;
        int cart_size;
        int sph_size;
    };

std::vector<int> compute_shell_offsets_cartesian(
    const std::vector<int>& angular_momenta
);

std::vector<int> compute_shell_offsets_spherical(
    const std::vector<int>& angular_momenta
);

    BasisTransformationHelper(const std::vector<int>& angular_momenta);

    const std::vector<ShellInfo>& get_shell_info() const { return shells_; }
    int get_total_cart_functions() const { return total_cart_; }
    int get_total_sph_functions() const { return total_sph_; }
    bool needs_transformation() const { return needs_transform_; }

private:
    std::vector<ShellInfo> shells_;
    int total_cart_;
    int total_sph_;
    bool needs_transform_;
};

std::vector<int> compute_shell_offsets_cartesian(
    const std::vector<int>& angular_momenta
);

std::vector<int> compute_shell_offsets_spherical(
    const std::vector<int>& angular_momenta
);

}

#endif
