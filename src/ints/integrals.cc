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

/**
 * @file src/ints/integrals.cc
 * @brief Integral Engine Implementation (LIBCINT TURBO BACKEND)
 */

#include "mshqc/ints/integrals.h"

// Wajib dibungkus extern "C" karena libcint murni ditulis dalam C
extern "C" {
#include <cint.h>
int cint1e_ovlp_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_kin_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_nuc_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
void CINTdel_optimizer(CINTOpt **opt);
int cint2c2e_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint3c2e_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
// [INJECTED_PRIORITAS_2E_ANALYTIC_GRAD]
int cint1e_ipovlp_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_ipkin_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_ipnuc_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint1e_iprinv_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint2c2e_ip1_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint3c2e_ip1_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
int cint3c2e_ip2_sph(double *buf, int *shls, int *atm, int natm, int *bas, int nbas, double *env, CINTOpt *opt);
void cint2c2e_ip1_sph_optimizer(CINTOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
void cint3c2e_ip1_sph_optimizer(CINTOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
void cint3c2e_ip2_sph_optimizer(CINTOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
void cint2c2e_sph_optimizer(CINTOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
void cint3c2e_sph_optimizer(CINTOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
}
#include <iostream>
#include <cmath>
#include <algorithm>

#include <string>

#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <array>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mshqc {

IntegralEngine::IntegralEngine(const Molecule& mol, const BasisSet& basis)
    : mol_(mol), basis_(basis), nbasis_(basis.n_basis_functions()),
      opt_(nullptr), opt_2c2e_(nullptr), opt_3c2e_(nullptr) {

    convert_basis_to_libcint();
    CINTOpt* tmp_opt = nullptr;
    CINTOpt* tmp_opt_2c = nullptr;
    CINTOpt* tmp_opt_3c = nullptr;

    cint2e_sph_optimizer(&tmp_opt, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data());
    cint2c2e_sph_optimizer(&tmp_opt_2c, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data());
    cint3c2e_sph_optimizer(&tmp_opt_3c, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data());

    opt_      = static_cast<void*>(tmp_opt);
    opt_2c2e_ = static_cast<void*>(tmp_opt_2c);
    opt_3c2e_ = static_cast<void*>(tmp_opt_3c);

    cache_valid = false;
}

IntegralEngine::~IntegralEngine() {
    auto free_opt = [](void*& p) {
        if (p != nullptr) {
            CINTOpt* tmp = static_cast<CINTOpt*>(p);
            CINTdel_optimizer(&tmp);
            p = nullptr;
        }
    };
    free_opt(opt_);
    free_opt(opt_2c2e_);
    free_opt(opt_3c2e_);
}

// --- Gunakan opt_2c2e_ dan opt_3c2e_ di pemanggilan libcint ---
int IntegralEngine::compute_2c2e_block_raw(int sh_P, int sh_Q, double* buf_out) const {
    int shls[2] = {sh_P, sh_Q};
    return cint2c2e_sph(buf_out, shls,
                        const_cast<int*>(atm_.data()), mol_.n_atoms(),
                        const_cast<int*>(bas_.data()), basis_.n_shells(),
                        const_cast<double*>(env_.data()),
                        static_cast<CINTOpt*>(opt_2c2e_));
}

int IntegralEngine::compute_3c2e_block_raw(int sh_i, int sh_j, int sh_P, double* buf_out) const {
    int shls[3] = {sh_i, sh_j, sh_P};
    return cint3c2e_sph(buf_out, shls,
                        const_cast<int*>(atm_.data()), mol_.n_atoms(),
                        const_cast<int*>(bas_.data()), basis_.n_shells(),
                        const_cast<double*>(env_.data()),
                        static_cast<CINTOpt*>(opt_3c2e_));
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

    // 1. Pack Atoms
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

// ============================================================================
// ONE-ELECTRON INTEGRALS
// ============================================================================
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
                // [FIX 2] Tambahkan nullptr
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

// ============================================================================
// COMPUTE ERI DENSE (TWO-ELECTRON)
// ============================================================================
// [INJECTED_ABELIAN_8FOLD_PETITE_LIST]
namespace {

inline int abelian_sph_sign(int l, int m, int sx, int sy, int sz) {
    if (l == 0) return 1;
    if (l == 1) {
        static constexpr int idx_map[3] = {0, 1, 2};
        const int s[3] = {sx, sy, sz};
        return (m >= 0 && m < 3) ? s[idx_map[m]] : 1;
    }
    if (l == 2) {
        switch (m) {
            case 0: return sx * sy;
            case 1: return sy * sz;
            case 2: return 1;
            case 3: return sx * sz;
            case 4: return 1;
            default: return 1;
        }
    }
    if (l == 3) {
        switch (m) {
            case 0: return sy;
            case 1: return sx * sy * sz;
            case 2: return sy;
            case 3: return sz;
            case 4: return sx;
            case 5: return sz;
            case 6: return sx;
            default: return 1;
        }
    }
    if (l == 4) {
        switch (m) {
            case 0: return sx * sy;
            case 1: return sy * sz;
            case 2: return sx * sy;
            case 3: return sy * sz;
            case 4: return 1;
            case 5: return sx * sz;
            case 6: return 1;
            case 7: return sx * sz;
            case 8: return 1;
            default: return 1;
        }
    }
    return 1;
}

inline std::uint64_t canon_8fold_key(int a, int b, int c, int d) {
    if (a < b) std::swap(a, b);
    if (c < d) std::swap(c, d);
    std::uint64_t p1 = static_cast<std::uint64_t>(a) * (a + 1) / 2 + static_cast<std::uint64_t>(b);
    std::uint64_t p2 = static_cast<std::uint64_t>(c) * (c + 1) / 2 + static_cast<std::uint64_t>(d);
    if (p1 < p2) std::swap(p1, p2);
    return (p1 << 32) | p2;
}

struct AbelianShellOps {
    std::vector<std::array<int, 3>> signs;
    std::vector<std::vector<int>> shell_perm;
};

AbelianShellOps build_abelian_shell_ops(
    const Molecule& mol,
    const BasisSet& basis,
    const std::vector<int>& bas_arr)
{
    AbelianShellOps out;
    const int nat = static_cast<int>(mol.n_atoms());
    const int nbas = basis.n_shells();

    // Dukung harmonik sferis s, p, d, f, g (l <= 4)
    for (int s = 0; s < nbas; ++s) {
        if (bas_arr[s * BAS_SLOTS + ANG_OF] > 4) return out;
    }

    std::vector<std::vector<int>> atom_shells(nat);
    for (int s = 0; s < nbas; ++s) {
        const int a = bas_arr[s * BAS_SLOTS + ATOM_OF];
        if (a < 0 || a >= nat) return out;
        atom_shells[a].push_back(s);
    }

    static constexpr std::array<int, 3> kOps[8] = {
        { 1,  1,  1}, {-1, -1,  1}, {-1,  1, -1}, { 1, -1, -1},
        {-1, -1, -1}, { 1,  1, -1}, { 1, -1,  1}, {-1,  1,  1},
    };
    const double tol = 1.0e-4;

    for (const auto& sgn : kOps) {
        std::vector<int> aperm(nat, -1);
        std::vector<char> used(nat, 0);
        bool ok = true;
        for (int i = 0; i < nat; ++i) {
            const auto& ai = mol.atom(i);
            const double tx = sgn[0] * ai.x;
            const double ty = sgn[1] * ai.y;
            const double tz = sgn[2] * ai.z;
            int best = -1;
            double best_d = 1.0e30;
            for (int j = 0; j < nat; ++j) {
                const auto& aj = mol.atom(j);
                if (aj.atomic_number != ai.atomic_number) continue;
                const double dx = aj.x - tx, dy = aj.y - ty, dz = aj.z - tz;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < best_d) {
                    best_d = d;
                    best = j;
                }
            }
            if (best < 0 || best_d > tol || used[best]) {
                ok = false;
                break;
            }
            aperm[i] = best;
            used[best] = 1;
        }
        if (!ok) continue;

        std::vector<int> sperm(nbas, -1);
        for (int i = 0; i < nat && ok; ++i) {
            const int j = aperm[i];
            if (atom_shells[i].size() != atom_shells[j].size()) {
                ok = false;
                break;
            }
            for (size_t k = 0; k < atom_shells[i].size(); ++k) {
                const int si = atom_shells[i][k];
                const int sj = atom_shells[j][k];
                if (bas_arr[si * BAS_SLOTS + ANG_OF] != bas_arr[sj * BAS_SLOTS + ANG_OF] ||
                    bas_arr[si * BAS_SLOTS + NPRIM_OF] != bas_arr[sj * BAS_SLOTS + NPRIM_OF]) {
                    ok = false;
                    break;
                }
                sperm[si] = sj;
            }
        }
        if (ok) {
            out.signs.push_back(sgn);
            out.shell_perm.push_back(std::move(sperm));
        }
    }
    return out;
}

} // namespace

void IntegralEngine::set_abelian_symmetry(bool enable) {
    if (use_abelian_symmetry_ != enable) {
        use_abelian_symmetry_ = enable;
        cache_valid = false;
    }
}

const Eigen::Tensor<double, 4>& IntegralEngine::compute_eri() {
    if (cache_valid) return cached_eri;

    // [INJECTED_HPC_V5_ERI_OOM_GUARD]
    {
        double mem_lim_gb = 8.0;
        if (const char* env_mem = std::getenv("MSHQC_MEMORY_GB")) {
            const double v = std::atof(env_mem);
            if (v >= 0.5) mem_lim_gb = v;
        }
        const double est_gb =
            static_cast<double>(nbasis_) * nbasis_ * nbasis_ * nbasis_ *
            static_cast<double>(sizeof(double)) / 1.0e9;
        if (est_gb > mem_lim_gb * 0.85) {
            throw std::runtime_error(
                "Exact 4-center ERI tensor requires " + std::to_string(est_gb) +
                " GB RAM (nbasis=" + std::to_string(nbasis_) +
                "), exceeding budget (" + std::to_string(mem_lim_gb * 0.85) +
                " GB). Use eri_method='df' or 'cholesky', or set MSHQC_MEMORY_GB.");
        }
    }

    cached_eri.resize(nbasis_, nbasis_, nbasis_, nbasis_);
    cached_eri.setZero();
    auto shell2bf = basis_.shell_to_basis_function_map();
    const int nbas = basis_.n_shells();

    std::vector<int> shell_dim(nbas), shell_l(nbas);
    for (int s = 0; s < nbas; ++s) {
        shell_dim[s] = CINTcgto_spheric(s, bas_.data());
        shell_l[s] = bas_[s * BAS_SLOTS + ANG_OF];
    }

    std::vector<double> schwarz_max(static_cast<size_t>(nbas) * nbas, 0.0);
    #pragma omp parallel
    {
        std::vector<double> buf(65536, 0.0);
        #pragma omp for schedule(dynamic)
        for (int s1 = 0; s1 < nbas; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                const int dim1 = shell_dim[s1];
                const int dim2 = shell_dim[s2];
                const size_t req_sz = static_cast<size_t>(dim1) * dim2 * dim1 * dim2;
                if (buf.size() < req_sz) buf.resize(req_sz);

                int shls[4] = {s1, s2, s1, s2};
                int has_val = cint2e_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(),
                                         bas_.data(), nbas, env_.data(), static_cast<CINTOpt*>(opt_));
                if (has_val) {
                    double max_v = 0.0;
                    for (size_t x = 0; x < req_sz; ++x) max_v = std::max(max_v, std::abs(buf[x]));
                    schwarz_max[static_cast<size_t>(s1) * nbas + s2] = std::sqrt(max_v);
                    schwarz_max[static_cast<size_t>(s2) * nbas + s1] = std::sqrt(max_v);
                }
            }
        }
    }

    AbelianShellOps abel;
    const bool active_abelian = use_abelian_symmetry_;
    if (active_abelian) {
        abel = build_abelian_shell_ops(mol_, basis_, bas_);
    }
    const int n_ops = static_cast<int>(abel.signs.size());
    const bool use_petite = (active_abelian && n_ops > 1);

    const double screen_thresh = 1e-12;
    size_t eval_quartets = 0;

    #pragma omp parallel reduction(+:eval_quartets)
    {
        std::vector<double> buf(65536, 0.0);
        #pragma omp for schedule(dynamic, 1)
        for (int s1 = 0; s1 < nbas; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                const double bound1 = schwarz_max[static_cast<size_t>(s1) * nbas + s2];
                if (bound1 < screen_thresh) continue;

                for (int s3 = 0; s3 <= s1; ++s3) {
                    const int s4_max = (s1 == s3) ? s2 : s3;
                    for (int s4 = 0; s4 <= s4_max; ++s4) {
                        if (bound1 * schwarz_max[static_cast<size_t>(s3) * nbas + s4] < screen_thresh) continue;

                        std::uint64_t my_key = 0;
                        if (use_petite) {
                            my_key = canon_8fold_key(s1, s2, s3, s4);
                            bool is_canon = true;
                            for (int g = 1; g < n_ops; ++g) {
                                const auto& sp = abel.shell_perm[g];
                                if (canon_8fold_key(sp[s1], sp[s2], sp[s3], sp[s4]) < my_key) {
                                    is_canon = false;
                                    break;
                                }
                            }
                            if (!is_canon) continue;
                        }

                        const int dim1 = shell_dim[s1];
                        const int dim2 = shell_dim[s2];
                        const int dim3 = shell_dim[s3];
                        const int dim4 = shell_dim[s4];
                        const size_t req_sz = static_cast<size_t>(dim1) * dim2 * dim3 * dim4;
                        if (buf.size() < req_sz) buf.resize(req_sz);

                        int shls[4] = {s1, s2, s3, s4};
                        eval_quartets += 1;
                        if (!cint2e_sph(buf.data(), shls, atm_.data(), mol_.n_atoms(),
                                        bas_.data(), nbas, env_.data(), static_cast<CINTOpt*>(opt_))) {
                            continue;
                        }

                        if (!use_petite) {
                            const int bf1 = shell2bf[s1], bf2 = shell2bf[s2];
                            const int bf3 = shell2bf[s3], bf4 = shell2bf[s4];
                            for (int i = 0; i < dim1; ++i) {
                                for (int j = 0; j < dim2; ++j) {
                                    for (int k = 0; k < dim3; ++k) {
                                        for (int l = 0; l < dim4; ++l) {
                                            const double val = buf[i + dim1 * (j + dim2 * (k + dim3 * l))];
                                            const Eigen::Index i1 = bf1 + i, i2 = bf2 + j;
                                            const Eigen::Index i3 = bf3 + k, i4 = bf4 + l;
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
                        } else {
                            std::uint64_t seen_keys[8];
                            int n_seen = 0;
                            const int l1 = shell_l[s1], l2 = shell_l[s2];
                            const int l3 = shell_l[s3], l4 = shell_l[s4];

                            for (int g = 0; g < n_ops; ++g) {
                                const auto& sp = abel.shell_perm[g];
                                const int t1 = sp[s1], t2 = sp[s2], t3 = sp[s3], t4 = sp[s4];
                                const std::uint64_t kg = (g == 0) ? my_key : canon_8fold_key(t1, t2, t3, t4);
                                bool dup = false;
                                for (int u = 0; u < n_seen; ++u) {
                                    if (seen_keys[u] == kg) {
                                        dup = true;
                                        break;
                                    }
                                }
                                if (dup) continue;
                                seen_keys[n_seen++] = kg;

                                const int sx = abel.signs[g][0], sy = abel.signs[g][1], sz = abel.signs[g][2];
                                const int bf1 = shell2bf[t1], bf2 = shell2bf[t2];
                                const int bf3 = shell2bf[t3], bf4 = shell2bf[t4];

                                int sg1[9], sg2[9], sg3[9], sg4[9];
                                for (int i = 0; i < dim1; ++i) sg1[i] = abelian_sph_sign(l1, i, sx, sy, sz);
                                for (int j = 0; j < dim2; ++j) sg2[j] = abelian_sph_sign(l2, j, sx, sy, sz);
                                for (int k = 0; k < dim3; ++k) sg3[k] = abelian_sph_sign(l3, k, sx, sy, sz);
                                for (int l = 0; l < dim4; ++l) sg4[l] = abelian_sph_sign(l4, l, sx, sy, sz);

                                for (int i = 0; i < dim1; ++i) {
                                    const Eigen::Index i1 = bf1 + i;
                                    const int p1 = sg1[i];
                                    for (int j = 0; j < dim2; ++j) {
                                        const Eigen::Index i2 = bf2 + j;
                                        const int p12 = p1 * sg2[j];
                                        for (int k = 0; k < dim3; ++k) {
                                            const Eigen::Index i3 = bf3 + k;
                                            const int p123 = p12 * sg3[k];
                                            for (int l = 0; l < dim4; ++l) {
                                                const double val = static_cast<double>(p123 * sg4[l]) *
                                                    buf[i + dim1 * (j + dim2 * (k + dim3 * l))];
                                                const Eigen::Index i4 = bf4 + l;
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
        }
    }

    last_eri_quartets_ = eval_quartets;
    cache_valid = true;
    return cached_eri;
}

// ============================================================================
// SINGLE SHELL BLOCK COMPUTE (THREAD-SAFE)
// ============================================================================
std::vector<double> IntegralEngine::compute_shell_block(int sh_a, int sh_b, int sh_c, int sh_d) {
    // Alokasi memori lokal per pemanggilan fungsi, mengeliminasi thread_local
    // untuk mencegah race condition pada akses OpenMP konkuren.
    std::vector<double> t_buffer;
    
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

int IntegralEngine::compute_shell_block_raw(int sh_a, int sh_b, int sh_c, int sh_d, double* buf_out) const {
    int shls[4] = {sh_a, sh_b, sh_c, sh_d};
    CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_);
    return cint2e_sph(buf_out, shls,
                      const_cast<int*>(atm_.data()), mol_.n_atoms(),
                      const_cast<int*>(bas_.data()), basis_.n_shells(),
                      const_cast<double*>(env_.data()), tmp_opt);
}

const double* IntegralEngine::compute_shell_block_ptr(int sh_a, int sh_b, int sh_c, int sh_d, size_t& size_out) {
    static thread_local std::vector<double> tl_buf;
    tl_buf = compute_shell_block(sh_a, sh_b, sh_c, sh_d);
    size_out = tl_buf.size();
    return tl_buf.data();
}
// ============================================================================
// DENSITY FITTING INTEGRALS (2-Center & 3-Center)
// ============================================================================

std::vector<double> IntegralEngine::compute_2c2e_block(int sh_P, int sh_Q) {
    std::vector<double> t_buffer;
    
    int dimP = CINTcgto_spheric(sh_P, bas_.data());
    int dimQ = CINTcgto_spheric(sh_Q, bas_.data());
    size_t sz = dimP * dimQ;
    
    if (t_buffer.size() < sz) t_buffer.resize(sz);
    
    int shls[2] = {sh_P, sh_Q};
    // [INJECTED_FINAL_OPT_CINTOPT_VECTORS]
    CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_2c2e_);
    int has_val = cint2c2e_sph(t_buffer.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data(), tmp_opt);
    
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
    CINTOpt* tmp_opt = static_cast<CINTOpt*>(opt_3c2e_);
    int has_val = cint3c2e_sph(t_buffer.data(), shls, atm_.data(), mol_.n_atoms(), bas_.data(), basis_.n_shells(), env_.data(), tmp_opt);
    
    if (!has_val) {
        std::fill(t_buffer.begin(), t_buffer.begin() + sz, 0.0);
    }
    
    return t_buffer;
}






// [INJECTED_PRIORITAS_2E_ANALYTIC_GRAD]
Eigen::MatrixXd IntegralEngine::compute_1e_gradient(
    const Eigen::MatrixXd& P_tot_ao,
    const Eigen::MatrixXd& W_eff_ao) const
{
    const int natm = static_cast<int>(mol_.n_atoms());
    const int nbas = basis_.n_shells();
    auto shell2bf = basis_.shell_to_basis_function_map();

    int max_dim = 1;
    for (int s = 0; s < nbas; ++s) {
        max_dim = std::max(max_dim, CINTcgto_spheric(s, bas_.data()));
    }
    const size_t buf_sz = std::max<size_t>(64, static_cast<size_t>(max_dim) * max_dim * 3);

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natm, 3);

    #pragma omp parallel
    {
        Eigen::MatrixXd g_loc = Eigen::MatrixXd::Zero(natm, 3);
        std::vector<double> buf_s(buf_sz), buf_t(buf_sz), buf_v(buf_sz);

        #pragma omp for schedule(dynamic, 4) nowait
        for (int s1 = 0; s1 < nbas; ++s1) {
            const int atm1 = bas_[s1 * BAS_SLOTS + ATOM_OF];
            const int dim1 = CINTcgto_spheric(s1, bas_.data());
            const int bf1  = shell2bf[s1];

            for (int s2 = 0; s2 < nbas; ++s2) {
                const int dim2 = CINTcgto_spheric(s2, bas_.data());
                const int bf2  = shell2bf[s2];
                int shls[2] = {s1, s2};

                int has_s = cint1e_ipovlp_sph(buf_s.data(), shls, const_cast<int*>(atm_.data()), natm,
                                              const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()), nullptr);
                int has_t = cint1e_ipkin_sph(buf_t.data(), shls, const_cast<int*>(atm_.data()), natm,
                                             const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()), nullptr);
                int has_v = cint1e_ipnuc_sph(buf_v.data(), shls, const_cast<int*>(atm_.data()), natm,
                                             const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()), nullptr);

                for (int xyz = 0; xyz < 3; ++xyz) {
                    const size_t comp_off = static_cast<size_t>(xyz) * dim1 * dim2;
                    double acc_s = 0.0, acc_tv = 0.0;
                    for (int j = 0; j < dim2; ++j) {
                        const double* p_col = P_tot_ao.col(bf2 + j).data() + bf1;
                        const double* w_col = W_eff_ao.col(bf2 + j).data() + bf1;
                        const size_t base = comp_off + static_cast<size_t>(j) * dim1;
                        for (int i = 0; i < dim1; ++i) {
                            if (has_s) acc_s += w_col[i] * buf_s[base + i];
                            double tv = (has_t ? buf_t[base + i] : 0.0) + (has_v ? buf_v[base + i] : 0.0);
                            acc_tv += p_col[i] * tv;
                        }
                    }
                    g_loc(atm1, xyz) += 2.0 * acc_s - 2.0 * acc_tv;
                }
            }
        }
        #pragma omp critical
        { grad.noalias() += g_loc; }
    }

    #pragma omp parallel
    {
        std::vector<double> env_loc = env_;
        std::vector<double> buf_r(buf_sz);

        #pragma omp for schedule(dynamic, 1) nowait
        for (int a = 0; a < natm; ++a) {
            const double Z_a = static_cast<double>(mol_.atom(a).atomic_number);
            env_loc[PTR_RINV_ORIG + 0] = mol_.atom(a).x;
            env_loc[PTR_RINV_ORIG + 1] = mol_.atom(a).y;
            env_loc[PTR_RINV_ORIG + 2] = mol_.atom(a).z;

            double g_a[3] = {0.0, 0.0, 0.0};
            for (int s1 = 0; s1 < nbas; ++s1) {
                const int dim1 = CINTcgto_spheric(s1, bas_.data());
                const int bf1  = shell2bf[s1];
                for (int s2 = 0; s2 < nbas; ++s2) {
                    const int dim2 = CINTcgto_spheric(s2, bas_.data());
                    const int bf2  = shell2bf[s2];
                    int shls[2] = {s1, s2};
                    if (!cint1e_iprinv_sph(buf_r.data(), shls, const_cast<int*>(atm_.data()), natm,
                                           const_cast<int*>(bas_.data()), nbas, env_loc.data(), nullptr)) continue;
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        const size_t comp_off = static_cast<size_t>(xyz) * dim1 * dim2;
                        double acc = 0.0;
                        for (int j = 0; j < dim2; ++j) {
                            const double* p_col = P_tot_ao.col(bf2 + j).data() + bf1;
                            const size_t base = comp_off + static_cast<size_t>(j) * dim1;
                            for (int i = 0; i < dim1; ++i) acc += p_col[i] * buf_r[base + i];
                        }
                        g_a[xyz] -= 2.0 * Z_a * acc;
                    }
                }
            }
            for (int xyz = 0; xyz < 3; ++xyz) grad(a, xyz) += g_a[xyz];
        }
    }

    return grad;
}

Eigen::MatrixXd IntegralEngine::compute_df_2e_gradient(
    int n_shells_prim,
    int n_shells_aux,
    int n_prim_bf,
    int n_aux_bf,
    const Eigen::MatrixXd& Gamma_tilde_3c,
    const Eigen::MatrixXd& gamma_tilde_2c) const
{
    (void)n_aux_bf;
    const int natm = static_cast<int>(mol_.n_atoms());
    const int nbas = basis_.n_shells();
    auto shell2bf = basis_.shell_to_basis_function_map();

    std::vector<int> prim_bf(n_shells_prim), prim_dim(n_shells_prim);
    int max_p = 1;
    for (int s = 0; s < n_shells_prim; ++s) {
        prim_bf[s]  = shell2bf[s];
        prim_dim[s] = CINTcgto_spheric(s, bas_.data());
        max_p = std::max(max_p, prim_dim[s]);
    }
    std::vector<int> aux_bf(n_shells_aux), aux_dim(n_shells_aux);
    int max_a = 1;
    int off_a = 0;
    for (int r = 0; r < n_shells_aux; ++r) {
        aux_bf[r]  = off_a;
        aux_dim[r] = CINTcgto_spheric(n_shells_prim + r, bas_.data());
        max_a = std::max(max_a, aux_dim[r]);
        off_a += aux_dim[r];
    }

    CINTOpt *opt_2c_ip1 = nullptr, *opt_3c_ip1 = nullptr, *opt_3c_ip2 = nullptr;
    cint2c2e_ip1_sph_optimizer(&opt_2c_ip1, const_cast<int*>(atm_.data()), natm,
                               const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()));
    cint3c2e_ip1_sph_optimizer(&opt_3c_ip1, const_cast<int*>(atm_.data()), natm,
                               const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()));
    cint3c2e_ip2_sph_optimizer(&opt_3c_ip2, const_cast<int*>(atm_.data()), natm,
                               const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()));

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natm, 3);
    const size_t buf_3c_sz = std::max<size_t>(64, static_cast<size_t>(max_p) * max_p * max_a * 3);
    const size_t buf_2c_sz = std::max<size_t>(64, static_cast<size_t>(max_a) * max_a * 3);

  // [INJECTED_AUDIT_FIX_DF_2E_GRAD_SCREEN]
  auto block_has_density = [&](int bf1, int dim1, int bf2, int dim2, int bfR, int dimR) -> bool {
    for (int p = 0; p < dimR; ++p) {
      const double* g_col = Gamma_tilde_3c.col(bfR + p).data();
      for (int j = 0; j < dim2; ++j) {
        const int row_base = (bf2 + j) * n_prim_bf + bf1;
        for (int i = 0; i < dim1; ++i) {
          if (std::abs(g_col[row_base + i]) > 1e-14) return true;
        }
      }
    }
    return false;
  };

    #pragma omp parallel
    {
        Eigen::MatrixXd g_loc = Eigen::MatrixXd::Zero(natm, 3);
        std::vector<double> buf(buf_3c_sz);

        #pragma omp for schedule(dynamic, 1) nowait
        for (int s1 = 0; s1 < n_shells_prim; ++s1) {
            const int atm1 = bas_[s1 * BAS_SLOTS + ATOM_OF];
            const int dim1 = prim_dim[s1], bf1 = prim_bf[s1];

            for (int s2 = 0; s2 < n_shells_prim; ++s2) {
                const int dim2 = prim_dim[s2], bf2 = prim_bf[s2];
                for (int r = 0; r < n_shells_aux; ++r) {
                    const int abs_r = n_shells_prim + r;
                    const int dimR = aux_dim[r], bfR = aux_bf[r];
                    if (!block_has_density(bf1, dim1, bf2, dim2, bfR, dimR)) continue;
          int shls[3] = {s1, s2, abs_r};
                    if (!cint3c2e_ip1_sph(buf.data(), shls, const_cast<int*>(atm_.data()), natm,
                                          const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()), opt_3c_ip1)) continue;

                    const size_t block_sz = static_cast<size_t>(dim1) * dim2 * dimR;
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        const double* b_xyz = buf.data() + static_cast<size_t>(xyz) * block_sz;
                        double acc = 0.0;
                        for (int p = 0; p < dimR; ++p) {
                            const double* g_col = Gamma_tilde_3c.col(bfR + p).data();
                            const size_t p_off = static_cast<size_t>(p) * dim1 * dim2;
                            for (int j = 0; j < dim2; ++j) {
                                const int row_base = (bf2 + j) * n_prim_bf + bf1;
                                const size_t jp_off = p_off + static_cast<size_t>(j) * dim1;
                                for (int i = 0; i < dim1; ++i) {
                                    acc += g_col[row_base + i] * b_xyz[jp_off + i];
                                }
                            }
                        }
                        g_loc(atm1, xyz) -= 2.0 * acc;
                    }
                }
            }
        }

        #pragma omp for schedule(dynamic, 1) nowait
        for (int s1 = 0; s1 < n_shells_prim; ++s1) {
            const int dim1 = prim_dim[s1], bf1 = prim_bf[s1];
            for (int s2 = 0; s2 <= s1; ++s2) {
                const int dim2 = prim_dim[s2], bf2 = prim_bf[s2];
                const double sym_fac = (s1 == s2) ? 1.0 : 2.0;

                for (int r = 0; r < n_shells_aux; ++r) {
                    const int abs_r = n_shells_prim + r;
                    const int atmR = bas_[abs_r * BAS_SLOTS + ATOM_OF];
                    const int dimR = aux_dim[r], bfR = aux_bf[r];
                    if (!block_has_density(bf1, dim1, bf2, dim2, bfR, dimR)) continue;
          int shls[3] = {s1, s2, abs_r};
                    if (!cint3c2e_ip2_sph(buf.data(), shls, const_cast<int*>(atm_.data()), natm,
                                          const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()), opt_3c_ip2)) continue;

                    const size_t block_sz = static_cast<size_t>(dim1) * dim2 * dimR;
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        const double* b_xyz = buf.data() + static_cast<size_t>(xyz) * block_sz;
                        double acc = 0.0;
                        for (int p = 0; p < dimR; ++p) {
                            const double* g_col = Gamma_tilde_3c.col(bfR + p).data();
                            const size_t p_off = static_cast<size_t>(p) * dim1 * dim2;
                            for (int j = 0; j < dim2; ++j) {
                                const int row_base = (bf2 + j) * n_prim_bf + bf1;
                                const size_t jp_off = p_off + static_cast<size_t>(j) * dim1;
                                for (int i = 0; i < dim1; ++i) {
                                    acc += g_col[row_base + i] * b_xyz[jp_off + i];
                                }
                            }
                        }
                        g_loc(atmR, xyz) -= sym_fac * acc;
                    }
                }
            }
        }

        std::vector<double> buf2(buf_2c_sz);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int r = 0; r < n_shells_aux; ++r) {
            const int abs_r = n_shells_prim + r;
            const int atmR = bas_[abs_r * BAS_SLOTS + ATOM_OF];
            const int dimR = aux_dim[r], bfR = aux_bf[r];
            for (int s = 0; s < n_shells_aux; ++s) {
                const int abs_s = n_shells_prim + s;
                const int dimS = aux_dim[s], bfS = aux_bf[s];
                int shls[2] = {abs_r, abs_s};
                if (!cint2c2e_ip1_sph(buf2.data(), shls, const_cast<int*>(atm_.data()), natm,
                                      const_cast<int*>(bas_.data()), nbas, const_cast<double*>(env_.data()), opt_2c_ip1)) continue;

                const size_t block_sz = static_cast<size_t>(dimR) * dimS;
                for (int xyz = 0; xyz < 3; ++xyz) {
                    const double* b_xyz = buf2.data() + static_cast<size_t>(xyz) * block_sz;
                    double acc = 0.0;
                    for (int q = 0; q < dimS; ++q) {
                        const double* gm_col = gamma_tilde_2c.col(bfS + q).data() + bfR;
                        const size_t q_off = static_cast<size_t>(q) * dimR;
                        for (int p = 0; p < dimR; ++p) {
                            acc += gm_col[p] * b_xyz[q_off + p];
                        }
                    }
                    g_loc(atmR, xyz) += acc;
                }
            }
        }

        #pragma omp critical
        { grad.noalias() += g_loc; }
    }

    CINTdel_optimizer(&opt_2c_ip1);
    CINTdel_optimizer(&opt_3c_ip1);
    CINTdel_optimizer(&opt_3c_ip2);
    return grad;
}

// STUBS
Eigen::Tensor<double, 3> IntegralEngine::compute_3center_eri(const BasisSet&) { return Eigen::Tensor<double, 3>(1,1,1); }
Eigen::MatrixXd IntegralEngine::compute_2center_eri(const BasisSet&) { return Eigen::MatrixXd::Zero(1,1); }
double IntegralEngine::compute_single_eri(int, int, int, int) { return 0.0; }
Eigen::VectorXd IntegralEngine::compute_eri_diagonal() { return Eigen::VectorXd::Zero(nbasis_); }
Eigen::VectorXd IntegralEngine::compute_eri_column(int) { return Eigen::VectorXd::Zero(nbasis_); }

} // namespace mshqc
