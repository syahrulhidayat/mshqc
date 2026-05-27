/**
 * @file include/mshqc/soscf.h
 * @brief Preconditioned Conjugate Gradient (PCG) & Trust-Region Solver
 */

#ifndef MSHQC_SOSCF_H
#define MSHQC_SOSCF_H

#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <cmath>
#include <algorithm>
#ifdef I
#undef I
#endif

namespace mshqc {

struct SOSCF_Result {
    Eigen::VectorXd step;       // Vektor rotasi orbital final
    int micro_iterations;       // Jumlah langkah internal PCG
    bool hit_trust_region;      // True jika langkah menabrak batas maksimal rotasi
    bool negative_curvature;    // True jika menemukan jalur energi menurun tajam
};

class PCGSolver {
public:
    int max_micro_iter = 15;         // Maksimal iterasi dalam mencari matriks rotasi
    double tolerance = 1e-3;         // Batas konvergensi PCG
    double trust_radius = 0.35;      // Maksimal radian putaran orbital (sangat aman)
    int print_level = 0;

    /**
     * @brief Menyelesaikan persamaan linear H * x = -g menggunakan PCG
     * @param gradient Vektor gradien (g) dalam 1D
     * @param diag_hessian Vektor diagonal Hessian (untuk Preconditioner)
     * @param calc_Hv Fungsi untuk menghitung Hessian-Vector Product
     */
    SOSCF_Result solve(
        const Eigen::VectorXd& gradient,
        const Eigen::VectorXd& diag_hessian,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> calc_Hv) 
    {
        int n = gradient.size();
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);  // Tebakan rotasi awal (nol)
        Eigen::VectorXd r = -gradient;                 // Residual awal
        
        // 1. Bangun Preconditioner dari Diagonal Hessian
        Eigen::VectorXd M_inv = Eigen::VectorXd::Zero(n);
        for(int i = 0; i < n; ++i) {
            double val = diag_hessian(i);
            // Level Shift: Cegah pembagian dengan nol jika kurvatur datar
            if (val < 1e-3) val = 1e-3; 
            M_inv(i) = 1.0 / val;
        }

        Eigen::VectorXd z = M_inv.cwiseProduct(r);
        Eigen::VectorXd p = z;
        double r_z_old = r.dot(z);
        
        SOSCF_Result res;
        res.micro_iterations = 0;
        res.hit_trust_region = false;
        res.negative_curvature = false;

        double g_norm = gradient.norm();
        // Dynamic Tolerance: Makin dekat ke konvergensi makro, makin teliti PCG-nya
        double current_tol = std::min(tolerance, g_norm * 0.1); 

        // 2. Micro-Iterations (PCG Loop)
        for (int k = 0; k < max_micro_iter; ++k) {
            res.micro_iterations++;
            
            // Panggil Fungsi Hessian-Vector (O(N^5) akan terjadi di dalam callback ini nanti)
            Eigen::VectorXd Hp = calc_Hv(p);
            
            double p_Hp = p.dot(Hp);
            
            // Cek Kurvatur Negatif
            if (p_Hp <= 1e-14) {
                res.negative_curvature = true;
                if (k == 0) {
                    x = z; 
                    if (x.norm() > trust_radius) {
                        x = x.normalized() * trust_radius;
                        res.hit_trust_region = true;
                    }
                }
                break;
            }

            double alpha = r_z_old / p_Hp;
            Eigen::VectorXd x_next = x + alpha * p;

            // 3. Trust-Region Boundary Check (Potong langkah jika melebihi radius)
            if (x_next.norm() > trust_radius) {
                double a = p.dot(p);
                double b = 2.0 * x.dot(p);
                double c = x.dot(x) - trust_radius * trust_radius;
                double det = b*b - 4.0*a*c;
                
                if (det > 0.0) {
                    double tau = (-b + std::sqrt(det)) / (2.0 * a);
                    x += tau * p;
                } else {
                    x = x_next.normalized() * trust_radius;
                }
                res.hit_trust_region = true;
                break; 
            }

            x = x_next;
            r -= alpha * Hp;

            if (r.norm() < current_tol) break;

            z = M_inv.cwiseProduct(r);
            double r_z_new = r.dot(z);
            double beta = r_z_new / r_z_old;
            p = z + beta * p;
            r_z_old = r_z_new;
        }

        res.step = x;
        return res;
    }
};

} // namespace mshqc
#endif