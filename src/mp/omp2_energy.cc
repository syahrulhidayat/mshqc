#include "mshqc/mp2/mp2.h"
#include "mshqc/integrals/eri_transformer.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <omp.h>
#include <tblis/tblis.h>

namespace mshqc {

void OMP2::compute_t2_amplitudes() {
    t2_aa_.clear(); t2_bb_.clear(); t2_ab_.clear();
    int nf = n_frozen_;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

    auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
    auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

    for (const auto& o1 : occ_spaces_a) {
        if (o1.size == 0) continue; 
        for (const auto& v1 : vir_spaces_a) {
            if (v1.size == 0) continue; 
            for (const auto& o2 : occ_spaces_a) {
                if (o2.size == 0) continue; 
                for (const auto& v2 : vir_spaces_a) {
                    if (v2.size == 0) continue; 

                    if ((o1.id ^ v1.id ^ o2.id ^ v2.id) == 0) {
                        auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                        auto* g_blk_ex = g_aa_.get_block(o1.id, v2.id, o2.id, v1.id);

                        bool ex_valid = (g_blk_ex != nullptr && g_blk_ex->dimension(1) == (Eigen::Index)v2.size && g_blk_ex->dimension(3) == (Eigen::Index)v1.size);

                        if (g_blk && ex_valid) { 
                            t2_aa_.allocate_block(o1.id, v1.id, o2.id, v2.id, o1.size, v1.size, o2.size, v2.size);
                            auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);

                            if (t_blk) {              
                                t_blk->setZero(); 
                                for (int di = 0; di < o1.size; ++di) {
                                    int i_glb = o1.offset + di;
                                    if (i_glb < nf) continue; 
                                    for (int dj = 0; dj < o2.size; ++dj) {
                                        int j_glb = o2.offset + dj;
                                        if (j_glb < nf) continue; 

                                        double e_ij = scf_.orbital_energies_alpha(i_glb) + scf_.orbital_energies_alpha(j_glb);
                                        for (int da = 0; da < v1.size; ++da) {
                                            double den_a = e_ij - scf_.orbital_energies_alpha(na_ + v1.offset + da);
                                            for (int db = 0; db < v2.size; ++db) {
                                                double den = den_a - scf_.orbital_energies_alpha(na_ + v2.offset + db);
                                                
                                                double val = (*g_blk)(di, da, dj, db);
                                                if (!is_restricted) val -= (*g_blk_ex)(di, db, dj, da);

                                                if (std::abs(den) < 1e-12) {
                                                    (*t_blk)(di, da, dj, db) = 0.0; 
                                                } else {
                                                    (*t_blk)(di, da, dj, db) = val / den; 
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
    }

    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        auto* g_bb_blk = g_bb_.get_block(0,0,0,0);
        if (g_bb_blk) {
            t2_bb_.allocate_block(0,0,0,0, nb_, nb_, vb_, vb_);
            auto* t_bb_blk = t2_bb_.get_block(0,0,0,0);
            if (t_bb_blk) t_bb_blk->setZero();

            for(int i = nf; i < nb_; ++i) for(int j = nf; j < nb_; ++j) {
                double e_ij = scf_.orbital_energies_beta(i) + scf_.orbital_energies_beta(j);
                for(int a=0; a<vb_; ++a) {
                    double den_a = e_ij - scf_.orbital_energies_beta(nb_+a);
                    for(int b=0; b<vb_; ++b) {
                        double den = den_a - scf_.orbital_energies_beta(nb_+b);
                        double val = (*g_bb_blk)(i, a, j, b) - (*g_bb_blk)(i, b, j, a);
                        double safe_den = (std::abs(den) < 1e-12) ? std::copysign(1e-12, den) : den;
                        (*t_bb_blk)(i, j, a, b) = val / safe_den;
                    }
                }
            }
        }

        auto* g_ab_blk = g_ab_.get_block(0,0,0,0);
        if (g_ab_blk) {
            t2_ab_.allocate_block(0,0,0,0, na_, nb_, va_, vb_);
            auto* t_ab_blk = t2_ab_.get_block(0,0,0,0);
            if (t_ab_blk) t_ab_blk->setZero(); 

            for(int i = nf; i < na_; ++i) for(int j = nf; j < nb_; ++j) {
                double e_ij = scf_.orbital_energies_alpha(i) + scf_.orbital_energies_beta(j);
                for(int a=0; a<va_; ++a) {
                    double den_a = e_ij - scf_.orbital_energies_alpha(na_+a);
                    for(int b=0; b<vb_; ++b) {
                        double den = den_a - scf_.orbital_energies_beta(nb_+b);
                        double safe_den = (std::abs(den) < 1e-12) ? std::copysign(1e-12, den) : den;
                        (*t_ab_blk)(i, j, a, b) = (*g_ab_blk)(i, a, j, b) / safe_den;
                    }
                }
            }
        }
    }
}
double OMP2::compute_mp2_energy() {
    double E_ss_aa = 0.0, E_ss_bb = 0.0, E_os = 0.0;
    int nf = n_frozen_;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);

    auto occ_spaces_a = get_irrep_spaces(scf_.irreps_alpha, 0, na_);
    auto vir_spaces_a = get_irrep_spaces(scf_.irreps_alpha, na_, va_);

    if (is_restricted) {
        double E_corr = 0.0;
        for (const auto& o1 : occ_spaces_a) {
            if (o1.size == 0) continue; 
            for (const auto& v1 : vir_spaces_a) {
                if (v1.size == 0) continue; 
                for (const auto& o2 : occ_spaces_a) {
                    if (o2.size == 0) continue; 
                    for (const auto& v2 : vir_spaces_a) {
                        if (v2.size == 0) continue; 

                        if ((o1.id ^ v1.id ^ o2.id ^ v2.id) != 0) continue;

                        auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                        auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                        auto* g_blk_ex = g_aa_.get_block(o1.id, v2.id, o2.id, v1.id);

                        bool ex_valid = (g_blk_ex != nullptr && 
                                         g_blk_ex->dimension(1) == (Eigen::Index)v2.size && 
                                         g_blk_ex->dimension(3) == (Eigen::Index)v1.size);

                        if (t_blk && g_blk && ex_valid) {
                            for (int di = 0; di < o1.size; ++di) {
                                if (o1.offset+di < nf) continue;
                                for (int dj = 0; dj < o2.size; ++dj) {
                                    if (o2.offset+dj < nf) continue;
                                    for (int da = 0; da < v1.size; ++da) {
                                        for (int db = 0; db < v2.size; ++db) {
                                            double g_dir = (*g_blk)(di, da, dj, db);
                                            double g_ex = (*g_blk_ex)(di, db, dj, da);
                                            double t_val = (*t_blk)(di, da, dj, db);
                                            E_corr += t_val * (2.0 * g_dir - g_ex);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        e_ss_ = 0.0;
        e_os_ = E_corr; 
        return E_corr;

    } else {
        for (const auto& o1 : occ_spaces_a) {
            if (o1.size == 0) continue; 
            for (const auto& v1 : vir_spaces_a) {
                if (v1.size == 0) continue; 
                for (const auto& o2 : occ_spaces_a) {
                    if (o2.size == 0) continue; 
                    for (const auto& v2 : vir_spaces_a) {
                        if (v2.size == 0) continue; 

                        if ((o1.id ^ v1.id ^ o2.id ^ v2.id) != 0) continue;

                        auto* t_blk = t2_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                        auto* g_blk = g_aa_.get_block(o1.id, v1.id, o2.id, v2.id);
                        auto* g_blk_ex = g_aa_.get_block(o1.id, v2.id, o2.id, v1.id);

                        bool ex_valid = false;
                        if (g_blk_ex != nullptr) {
                            if (g_blk_ex->dimension(1) == (Eigen::Index)v2.size && 
                                g_blk_ex->dimension(3) == (Eigen::Index)v1.size) {
                                ex_valid = true;
                            }
                        }

                        if (t_blk && g_blk && ex_valid) {
                            for (int di = 0; di < o1.size; ++di) {
                                if (o1.offset+di < nf) continue;
                                for (int dj = 0; dj < o2.size; ++dj) {
                                    if (o2.offset+dj < nf) continue;
                                    for (int da = 0; da < v1.size; ++da) {
                                        for (int db = 0; db < v2.size; ++db) {
                                            double g_val = (*g_blk)(di, da, dj, db) - (*g_blk_ex)(di, db, dj, da);
                                            E_ss_aa += (*t_blk)(di, da, dj, db) * g_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (nb_ > 0 && vb_ > 0) {
            auto* t_bb = t2_bb_.get_block(0,0,0,0);
            auto* g_bb = g_bb_.get_block(0,0,0,0);
            if (t_bb && g_bb) {
              #pragma omp parallel for collapse(2) reduction(+:E_ss_bb)
                for(int i=nf; i<nb_; ++i) for(int j=nf; j<nb_; ++j) for(int a=0; a<vb_; ++a) for(int b=0; b<vb_; ++b)
                    E_ss_bb += (*t_bb)(i, j, a, b) * ((*g_bb)(i, a, j, b) - (*g_bb)(i, b, j, a));
            }
            auto* t_ab = t2_ab_.get_block(0,0,0,0);
            auto* g_ab = g_ab_.get_block(0,0,0,0);
            if (t_ab && g_ab) {
                for(int i=nf; i<na_; ++i) for(int j=nf; j<nb_; ++j) for(int a=0; a<va_; ++a) for(int b=0; b<vb_; ++b)
                    E_os += (*t_ab)(i, j, a, b) * (*g_ab)(i, a, j, b);
            }
        }

        e_ss_ = 0.25 * E_ss_aa + 0.25 * E_ss_bb;
        e_os_ = E_os;
        return e_ss_ + e_os_;
    }
}
void OMP2::compute_t2_and_energy_cholesky() {
    double E_ss_aa = 0.0, E_ss_bb = 0.0, E_os = 0.0;
    int nf = n_frozen_;
    bool is_restricted = (na_ == nb_ && va_ == vb_ && mol_.multiplicity() == 1);
    
    // Pindahkan konstanta ke atas agar digunakan seragam oleh semua blok
    constexpr double sigma_sq = 1e-20; 

    t2_aa_.allocate_block(0, 0, 0, 0, na_, va_, na_, va_);
    auto* t_aa_blk = t2_aa_.get_block(0, 0, 0, 0);

    if (t_aa_blk) {
        #pragma omp parallel for reduction(+:E_ss_aa) schedule(dynamic, 1)
        for (int i = nf; i < na_; ++i) {
            Eigen::MatrixXd g_ijab(va_, va_); 

            for (int j = nf; j < na_; ++j) {
                Eigen::MatrixXd Bia = B_ia_P_alpha_.middleRows(i * va_, va_);
                Eigen::MatrixXd Bjb = B_ia_P_alpha_.middleRows(j * va_, va_);
                g_ijab.noalias() = Bia * Bjb.transpose(); 

                double e_ij = scf_.orbital_energies_alpha(i) + scf_.orbital_energies_alpha(j);

                for (int a = 0; a < va_; ++a) {
                    double den_a = e_ij - scf_.orbital_energies_alpha(na_ + a);
                    for (int b = 0; b < va_; ++b) {
                        double den = den_a - scf_.orbital_energies_alpha(na_ + b);

                        double val_dir = g_ijab(a, b);
                        double val_ex  = g_ijab(b, a); 

                        // Regularisasi mulus (Smooth Regularization)
                        double reg_den = den / (den * den + sigma_sq);
                        double t_val = 0.0;

                        if (is_restricted) {
                            t_val = val_dir * reg_den;
                            E_ss_aa += t_val * (2.0 * val_dir - val_ex); 
                        } else {
                            t_val = (val_dir - val_ex) * reg_den;
                            E_ss_aa += t_val * (val_dir - val_ex); 
                        }
                        
                        (*t_aa_blk)(i, a, j, b) = t_val;
                    }
                }
            }
        }
    }
    
    if (!is_restricted && nb_ > 0 && vb_ > 0) {
        t2_bb_.allocate_block(0, 0, 0, 0, nb_, nb_, vb_, vb_);
        auto* t_bb_blk = t2_bb_.get_block(0, 0, 0, 0);

        t2_ab_.allocate_block(0, 0, 0, 0, na_, nb_, va_, vb_);
        auto* t_ab_blk = t2_ab_.get_block(0, 0, 0, 0);

        // ========================================================
        // BLOK BETA-BETA
        // ========================================================
        if (t_bb_blk) {
            #pragma omp parallel for reduction(+:E_ss_bb) schedule(dynamic, 1)
            for (int i = nf; i < nb_; ++i) {
                Eigen::MatrixXd g_ijab(vb_, vb_);

                for (int j = nf; j < nb_; ++j) {
                    Eigen::MatrixXd Bia = B_ia_P_beta_.middleRows(i * vb_, vb_);
                    Eigen::MatrixXd Bjb = B_ia_P_beta_.middleRows(j * vb_, vb_);
                    g_ijab.noalias() = Bia * Bjb.transpose();

                    double e_ij = scf_.orbital_energies_beta(i) + scf_.orbital_energies_beta(j);

                    for (int a = 0; a < vb_; ++a) {
                        double den_a = e_ij - scf_.orbital_energies_beta(nb_ + a);
                        for (int b = 0; b < vb_; ++b) {
                            double den = den_a - scf_.orbital_energies_beta(nb_ + b);

                            double val_dir = g_ijab(a, b);
                            double val_ex  = g_ijab(b, a);
                            
                            // PERBAIKAN: Gunakan regularisasi sigma_sq, bukan std::copysign
                            double reg_den = den / (den * den + sigma_sq);
                            double t_val = (val_dir - val_ex) * reg_den;
                            
                            E_ss_bb += t_val * (val_dir - val_ex);
                            (*t_bb_blk)(i, j, a, b) = t_val; 
                        }
                    }
                }
            }
        }

        // ========================================================
        // BLOK ALPHA-BETA (Opposite Spin)
        // ========================================================
        if (t_ab_blk) {
            #pragma omp parallel for reduction(+:E_os) schedule(dynamic, 1)
            for (int i = nf; i < na_; ++i) {
                Eigen::MatrixXd g_ijab(va_, vb_);

                for (int j = nf; j < nb_; ++j) {
                    Eigen::MatrixXd Bia = B_ia_P_alpha_.middleRows(i * va_, va_);
                    Eigen::MatrixXd Bjb = B_ia_P_beta_.middleRows(j * vb_, vb_);
                    g_ijab.noalias() = Bia * Bjb.transpose(); 

                    double e_ij = scf_.orbital_energies_alpha(i) + scf_.orbital_energies_beta(j);

                    for (int a = 0; a < va_; ++a) {
                        double den_a = e_ij - scf_.orbital_energies_alpha(na_ + a);
                        for (int b = 0; b < vb_; ++b) {
                            double den = den_a - scf_.orbital_energies_beta(nb_ + b);

                            double val_dir = g_ijab(a, b);
                            
                            // PERBAIKAN: Gunakan regularisasi sigma_sq, bukan std::copysign
                            double reg_den = den / (den * den + sigma_sq);
                            double t_val = val_dir * reg_den;
                            
                            E_os += t_val * val_dir; 
                            (*t_ab_blk)(i, j, a, b) = t_val;
                        }
                    }
                }
            }
        }
    }

    if (is_restricted) {
        e_ss_ = 0.0;
        e_os_ = E_ss_aa;
    } else {
        e_ss_ = 0.25 * E_ss_aa + 0.25 * E_ss_bb;
        e_os_ = E_os;
    }
}
}