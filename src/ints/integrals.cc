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

#include "mshqc/ints/integrals.h"

extern "C" {
#include <cint.h>
int cint1e_ovlp_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_kin_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_nuc_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
void CINTdel_optimizer(CINTOpt **opt);
int cint2c2e_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint3c2e_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
}
#include <iostream>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {

IntegralEngine::IntegralEngine(const Molecule& mol, const BasisSet& basis)
    : mol_(mol), basis_(basis), nbasis_(basis.n_basis_functions()), opt_(nullptr) {

    convert_basis_to_libcint();
    CINTOpt* tmp_opt = nullptr;
    cint2e_sph_optimizer(&tmp_opt, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data());
    opt_ = static_cast<void*>(tmp_opt);

    cache_valid = false;
}

IntegralEngine::~IntegralEngine() {
    if (opt_ != nullptr) {
        CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_);
        CINTdel_optimizer(&tmp_opt);
    }
}

int IntegralEngine::find_atom_index(const std::array<double, 3>& center) {
    int best_i = 0;
    double min_dist = 1e10;
    for (size_t i = 0; i < mol_.n_atoms(); i++) {
        double dx = mol_.atom(i).x - center[0];
        double dy = mol_.atom(i).y - center[1];
        double dz = mol_.atom(i).z - center[2];
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < min_dist) {
            min_dist = dist;
            best_i = i;
        }
    }
    return best_i;
}
void IntegralEngine::convert_basis_to_libcint() {
    int natm = mol_.n_atoms();
    int nbas = basis_.n_shells();

    env_.assign(20, 0.0);
    atm_.resize(natm * ATM_SLOTS, 0);
    bas_.resize(nbas * BAS_SLOTS, 0);

    for (int i = 0; i < natm; ++i) {
        atm_[i * ATM_SLOTS + CHARGE_OF] = mol_.atom(i).atomic_number;
        atm_[i * ATM_SLOTS + PTR_COORD] = env_.size();
        env_.push_back(mol_.atom(i).x);
        env_.push_back(mol_.atom(i).y);
        env_.push_back(mol_.atom(i).z);
    }

    for (int s = 0; s < nbas; ++s) {
        const auto& shell = basis_.shell(s);
        auto pos = shell.position();
        bas_[s * BAS_SLOTS + ATOM_OF]  = find_atom_index({pos[0], pos[1], pos[2]});
        bas_[s * BAS_SLOTS + ANG_OF]   = shell.l();
        bas_[s * BAS_SLOTS + NPRIM_OF] = shell.n_primitives();
        bas_[s * BAS_SLOTS + NCTR_OF]  = 1;
        bas_[s * BAS_SLOTS + PTR_EXP] = env_.size();
        for (size_t p = 0; p < shell.n_primitives(); ++p) {
            env_.push_back(shell.primitive(p).exponent);
        }
        bas_[s * BAS_SLOTS + PTR_COEFF] = env_.size();
        for (size_t p = 0; p < shell.n_primitives(); ++p) {
            double exp = shell.primitive(p).exponent;
            double coef = shell.primitive(p).coefficient;
            double norm = CINTgto_norm(shell.l(), exp);
            env_.push_back(coef * norm);
        }
    }
}

Eigen::MatrixXd IntegralEngine::compute_overlap() {
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    auto shell2bf = basis_.shell_to_basis_function_map();
    int nbas = basis_.n_shells();

    #pragma omp parallel
    {
        std::vector<double> buf(10000);
        #pragma omp for schedule(dynamic)
        for (int s1 = 0; s1 < nbas; s1++) {
            for (int s2 = 0; s2 <= s1; s2++) {
                int shls[2] = {s1, s2};
                int has_val = cint1e_ovlp_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), nbas, env_.data(), nullptr);
                if (!has_val) continue;

                int dim1 = CINTcgto_spheric(s1, bas_.data());
                int dim2 = CINTcgto_spheric(s2, bas_.data());
                int bf1 = shell2bf[s1];
                int bf2 = shell2bf[s2];

                for (int i = 0; i < dim1; ++i) {
                    for (int j = 0; j < dim2; ++j) {
                        double val = buf[i + j * dim1];
                        S(bf1 + i, bf2 + j) = val;
                        S(bf2 + j, bf1 + i) = val;
                    }
                }
            }
        }
    }
    return S;
}

Eigen::MatrixXd IntegralEngine::compute_kinetic() {
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    auto shell2bf = basis_.shell_to_basis_function_map();
    int nbas = basis_.n_shells();

    #pragma omp parallel
    {
        std::vector<double> buf(10000);
        #pragma omp for schedule(dynamic)
        for (int s1 = 0; s1 < nbas; s1++) {
            for (int s2 = 0; s2 <= s1; s2++) {
                int shls[2] = {s1, s2};
                int has_val = cint1e_kin_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), nbas, env_.data(), nullptr);
                if (!has_val) continue;

                int dim1 = CINTcgto_spheric(s1, bas_.data());
                int dim2 = CINTcgto_spheric(s2, bas_.data());
                int bf1 = shell2bf[s1];
                int bf2 = shell2bf[s2];

                for (int i = 0; i < dim1; ++i) {
                    for (int j = 0; j < dim2; ++j) {
                        double val = buf[i + j * dim1];
                        T(bf1 + i, bf2 + j) = val;
                        T(bf2 + j, bf1 + i) = val;
                    }
                }
            }
        }
    }
    return T;
}

Eigen::MatrixXd IntegralEngine::compute_nuclear() {
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    auto shell2bf = basis_.shell_to_basis_function_map();
    int nbas = basis_.n_shells();

    #pragma omp parallel
    {
        std::vector<double> buf(10000);
        #pragma omp for schedule(dynamic)
        for (int s1 = 0; s1 < nbas; s1++) {
            for (int s2 = 0; s2 <= s1; s2++) {
                int shls[2] = {s1, s2};

                int has_val = cint1e_nuc_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), nbas, env_.data(), nullptr);
                if (!has_val) continue;

                int dim1 = CINTcgto_spheric(s1, bas_.data());
                int dim2 = CINTcgto_spheric(s2, bas_.data());
                int bf1 = shell2bf[s1];
                int bf2 = shell2bf[s2];

                for (int i = 0; i < dim1; ++i) {
                    for (int j = 0; j < dim2; ++j) {
                        double val = buf[i + j * dim1];
                        V(bf1 + i, bf2 + j) = val;
                        V(bf2 + j, bf1 + i) = val;
                    }
                }
            }
        }
    }
    return V;
}

Eigen::MatrixXd IntegralEngine::compute_core_hamiltonian() {
    return compute_kinetic() + compute_nuclear();
}

const Eigen::Tensor<double, 4>& IntegralEngine::compute_eri() {
    if (cache_valid) return cached_eri;

    cached_eri.resize(nbasis_, nbasis_, nbasis_, nbasis_);
    cached_eri.setZero();
    auto shell2bf = basis_.shell_to_basis_function_map();
    int nbas = basis_.n_shells();
    std::vector<double> schwarz_max(nbas * nbas, 0.0);
    #pragma omp parallel
    {
        std::vector<double> buf(10000);
        #pragma omp for schedule(dynamic)
        for (int s1 = 0; s1 < nbas; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                int shls[4] = {s1, s2, s1, s2};
                int has_val = cint2e_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), nbas, env_.data(), nullptr);
                if (has_val) {
                    int dim1 = CINTcgto_spheric(s1, bas_.data());
                    int dim2 = CINTcgto_spheric(s2, bas_.data());
                    double max_v = 0.0;
                    for (int x = 0; x < dim1*dim2*dim1*dim2; ++x) max_v = std::max(max_v, std::abs(buf[x]));
                    schwarz_max[s1 * nbas + s2] = std::sqrt(max_v);
                    schwarz_max[s2 * nbas + s1] = std::sqrt(max_v);
                }
            }
        }
    }

    const double screen_thresh = 1e-12;
    #pragma omp parallel
    {
        std::vector<double> buf(10000);
        #pragma omp for schedule(dynamic, 1)
        for (int s1 = 0; s1 < nbas; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                double bound1 = schwarz_max[s1 * nbas + s2];
                if (bound1 < screen_thresh) continue;

                for (int s3 = 0; s3 <= s1; ++s3) {
                    int s4_max = (s1 == s3) ? s2 : s3;
                    for (int s4 = 0; s4 <= s4_max; ++s4) {
                        if (bound1 * schwarz_max[s3 * nbas + s4] < screen_thresh) continue;

                        int shls[4] = {s1, s2, s3, s4};
                        if (!cint2e_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), nbas, env_.data(), nullptr)) continue;

                        int dim1 = CINTcgto_spheric(s1, bas_.data());
                        int dim2 = CINTcgto_spheric(s2, bas_.data());
                        int dim3 = CINTcgto_spheric(s3, bas_.data());
                        int dim4 = CINTcgto_spheric(s4, bas_.data());

                        int bf1 = shell2bf[s1]; int bf2 = shell2bf[s2];
                        int bf3 = shell2bf[s3]; int bf4 = shell2bf[s4];

                        for (int i = 0; i < dim1; ++i) {
                            for (int j = 0; j < dim2; ++j) {
                                for (int k = 0; k < dim3; ++k) {
                                    for (int l = 0; l < dim4; ++l) {
                                        double val = buf[i + dim1*(j + dim2*(k + dim3*l))];

                                        long i1 = bf1 + i, i2 = bf2 + j, i3 = bf3 + k, i4 = bf4 + l;
                                        cached_eri(i1, i2, i3, i4) = val;
                                        cached_eri(i2, i1, i3, i4) = val;
                                        cached_eri(i1, i2, i4, i3) = val;
                                        cached_eri(i2, i1, i4, i3) = val;
                                        cached_eri(i3, i4, i1, i2) = val;
                                        cached_eri(i4, i3, i1, i2) = val;
                                        cached_eri(i3, i4, i2, i1) = val;
                                        cached_eri(i4, i3, i2, i1) = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    cache_valid = true;
    return cached_eri;
}

const std::vector<double>& IntegralEngine::compute_shell_block(int sh_a, int sh_b, int sh_c, int sh_d) {
    thread_local std::vector<double> t_buffer;

    int dim1 = CINTcgto_spheric(sh_a, bas_.data());
    int dim2 = CINTcgto_spheric(sh_b, bas_.data());
    int dim3 = CINTcgto_spheric(sh_c, bas_.data());
    int dim4 = CINTcgto_spheric(sh_d, bas_.data());
    size_t sz = dim1 * dim2 * dim3 * dim4;

    if (t_buffer.size() < sz) t_buffer.resize(sz);

    int shls[4] = {sh_a, sh_b, sh_c, sh_d};

    CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_);
    int has_val = cint2e_sph(t_buffer.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data(), tmp_opt);

    if (!has_val) {
        std::fill(t_buffer.begin(), t_buffer.begin() + sz, 0.0);
    }

    return t_buffer;
}

const double* IntegralEngine::compute_shell_block_ptr(int sh_a, int sh_b, int sh_c, int sh_d, size_t& size_out) {
    const auto& buf = compute_shell_block(sh_a, sh_b, sh_c, sh_d);
    size_out = buf.size();
    return buf.data();
}

std::vector<double> IntegralEngine::compute_2c2e_block(int sh_P, int sh_Q) {
    std::vector<double> t_buffer;

    int dimP = CINTcgto_spheric(sh_P, bas_.data());
    int dimQ = CINTcgto_spheric(sh_Q, bas_.data());
    size_t sz = dimP * dimQ;

    if (t_buffer.size() < sz) t_buffer.resize(sz);

    int shls[2] = {sh_P, sh_Q};
    CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_);

    int has_val = cint2c2e_sph(t_buffer.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data(), nullptr);

    if (!has_val) {
        std::fill(t_buffer.begin(), t_buffer.begin() + sz, 0.0);
    }

    return t_buffer;
}

std::vector<double> IntegralEngine::compute_3c2e_block(int sh_i, int sh_j, int sh_P) {
    std::vector<double> t_buffer;

    int dim1 = CINTcgto_spheric(sh_i, bas_.data());
    int dim2 = CINTcgto_spheric(sh_j, bas_.data());
    int dimP = CINTcgto_spheric(sh_P, bas_.data());
    size_t sz = dim1 * dim2 * dimP;

    if (t_buffer.size() < sz) t_buffer.resize(sz);
    int shls[3] = {sh_i, sh_j, sh_P};
    CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_);
    int has_val = cint3c2e_sph(t_buffer.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data(), nullptr);

    if (!has_val) {
        std::fill(t_buffer.begin(), t_buffer.begin() + sz, 0.0);
    }

    return t_buffer;
}

Eigen::Tensor<double, 3> IntegralEngine::compute_3center_eri(const BasisSet&) { return Eigen::Tensor<double, 3>(1,1,1); }
Eigen::MatrixXd IntegralEngine::compute_2center_eri(const BasisSet&) { return Eigen::MatrixXd::Zero(1,1); }
double IntegralEngine::compute_single_eri(int, int, int, int) { return 0.0; }
Eigen::VectorXd IntegralEngine::compute_eri_diagonal() { return Eigen::VectorXd::Zero(nbasis_); }
Eigen::VectorXd IntegralEngine::compute_eri_column(int) { return Eigen::VectorXd::Zero(nbasis_); }

}
