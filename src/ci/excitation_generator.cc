/**
 * @file excitation_generator.cc
 * @brief Implementation of determinant excitation generators
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-14
 */

#include "mshqc/ci/excitation_generator.h"
#include <algorithm>

namespace mshqc {
namespace ci {

void generate_singles(const Determinant& det, 
                      int n_orb,
                      std::function<void(const GeneratedExcitation&)> callback) {
    
    auto occ_alpha = det.alpha_occupations();
    auto occ_beta = det.beta_occupations();
    
    // Alpha singles: i(α) → a(α)
    for (int i : occ_alpha) {
        for (int a = 0; a < n_orb; a++) {
            // Check if 'a' is virtual (not occupied)
            if (std::find(occ_alpha.begin(), occ_alpha.end(), a) == occ_alpha.end()) {
                Determinant excited = det.single_excite(i, a, true);  // true = alpha
                
                GeneratedExcitation exc;
                exc.det = excited;
                exc.exc_level = 1;
                exc.i = i;
                exc.j = -1;
                exc.a = a;
                exc.b = -1;
                exc.spin_i = true;  // alpha
                exc.spin_j = false;
                
                callback(exc);
            }
        }
    }
    
    // Beta singles: i(β) → a(β)
    for (int i : occ_beta) {
        for (int a = 0; a < n_orb; a++) {
            if (std::find(occ_beta.begin(), occ_beta.end(), a) == occ_beta.end()) {
                Determinant excited = det.single_excite(i, a, false);  // false = beta
                
                GeneratedExcitation exc;
                exc.det = excited;
                exc.exc_level = 1;
                exc.i = i;
                exc.j = -1;
                exc.a = a;
                exc.b = -1;
                exc.spin_i = false;  // beta
                exc.spin_j = false;
                
                callback(exc);
            }
        }
    }
}

void generate_doubles(const Determinant& det,
                      int n_orb,
                      std::function<void(const GeneratedExcitation&)> callback) {
    
    auto occ_alpha = det.alpha_occupations();
    auto occ_beta = det.beta_occupations();
    
    // Helper to check if orbital is virtual
    auto is_virtual_alpha = [&](int orb) {
        return std::find(occ_alpha.begin(), occ_alpha.end(), orb) == occ_alpha.end();
    };
    auto is_virtual_beta = [&](int orb) {
        return std::find(occ_beta.begin(), occ_beta.end(), orb) == occ_beta.end();
    };
    
    // Alpha-Alpha doubles: i(α),j(α) → a(α),b(α)
    for (size_t ii = 0; ii < occ_alpha.size(); ii++) {
        for (size_t jj = ii + 1; jj < occ_alpha.size(); jj++) {
            int i = occ_alpha[ii];
            int j = occ_alpha[jj];
            
            for (int a = 0; a < n_orb; a++) {
                if (!is_virtual_alpha(a)) continue;
                
                for (int b = a + 1; b < n_orb; b++) {
                    if (!is_virtual_alpha(b)) continue;
                    
                    Determinant excited = det.double_excite(i, j, a, b, true, true);
                    
                    GeneratedExcitation exc;
                    exc.det = excited;
                    exc.exc_level = 2;
                    exc.i = i;
                    exc.j = j;
                    exc.a = a;
                    exc.b = b;
                    exc.spin_i = true;  // alpha
                    exc.spin_j = true;  // alpha
                    
                    callback(exc);
                }
            }
        }
    }
    
    // Beta-Beta doubles: i(β),j(β) → a(β),b(β)
    for (size_t ii = 0; ii < occ_beta.size(); ii++) {
        for (size_t jj = ii + 1; jj < occ_beta.size(); jj++) {
            int i = occ_beta[ii];
            int j = occ_beta[jj];
            
            for (int a = 0; a < n_orb; a++) {
                if (!is_virtual_beta(a)) continue;
                
                for (int b = a + 1; b < n_orb; b++) {
                    if (!is_virtual_beta(b)) continue;
                    
                    Determinant excited = det.double_excite(i, j, a, b, false, false);
                    
                    GeneratedExcitation exc;
                    exc.det = excited;
                    exc.exc_level = 2;
                    exc.i = i;
                    exc.j = j;
                    exc.a = a;
                    exc.b = b;
                    exc.spin_i = false;  // beta
                    exc.spin_j = false;  // beta
                    
                    callback(exc);
                }
            }
        }
    }
    
    // Alpha-Beta doubles: i(α),j(β) → a(α),b(β)
    for (int i : occ_alpha) {
        for (int j : occ_beta) {
            for (int a = 0; a < n_orb; a++) {
                if (!is_virtual_alpha(a)) continue;
                
                for (int b = 0; b < n_orb; b++) {
                    if (!is_virtual_beta(b)) continue;
                    
                    Determinant excited = det.double_excite(i, j, a, b, true, false);
                    
                    GeneratedExcitation exc;
                    exc.det = excited;
                    exc.exc_level = 2;
                    exc.i = i;
                    exc.j = j;
                    exc.a = a;
                    exc.b = b;
                    exc.spin_i = true;   // alpha
                    exc.spin_j = false;  // beta
                    
                    callback(exc);
                }
            }
        }
    }
}

void generate_connected_excitations(const Determinant& det,
                                    int n_orb,
                                    std::function<void(const GeneratedExcitation&)> callback) {
    generate_singles(det, n_orb, callback);
    generate_doubles(det, n_orb, callback);
}

int count_connected_excitations(const Determinant& det, int n_orb) {
    auto occ_alpha = det.alpha_occupations();
    auto occ_beta = det.beta_occupations();
    
    int nocc_a = occ_alpha.size();
    int nocc_b = occ_beta.size();
    int nvirt_a = n_orb - nocc_a;
    int nvirt_b = n_orb - nocc_b;
    
    // Singles
    int n_singles = nocc_a * nvirt_a + nocc_b * nvirt_b;
    
    // Doubles
    int n_aa = (nocc_a * (nocc_a - 1) / 2) * (nvirt_a * (nvirt_a - 1) / 2);
    int n_bb = (nocc_b * (nocc_b - 1) / 2) * (nvirt_b * (nvirt_b - 1) / 2);
    int n_ab = nocc_a * nocc_b * nvirt_a * nvirt_b;
    int n_doubles = n_aa + n_bb + n_ab;
    
    return n_singles + n_doubles;
}

} // namespace ci
} // namespace mshqc
