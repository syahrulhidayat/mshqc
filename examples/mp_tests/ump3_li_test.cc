/**
 * @file ump3_li_test.cc
 * @brief Test UMP3 on Li atom (open-shell, doublet)
 *
 * Uses UHF -> UMP2 -> UMP3 (approximate) to get correlation increments.
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "====================================\n";
    std::cout << "  UMP3 Test: Li/cc-pVTZ\n";
    std::cout << "====================================\n";

    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);  // Li

    BasisSet basis("cc-pVTZ", li);
    std::cout << "Basis: cc-pVTZ (" << basis.n_basis_functions() << " functions)\n";

    auto integrals = std::make_shared<IntegralEngine>(li, basis);

    // UHF
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;

    int n_alpha = 2;
    int n_beta = 1;
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "UHF energy:   " << uhf_result.energy_total << " Ha\n";

    // UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();

    std::cout << std::setprecision(10);
    std::cout << "MP2 corr:     " << ump2_result.e_corr_total << " Ha\n";

    // UMP3 (approximate)
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();

    std::cout << "MP3 corr:     " << ump3_result.e_mp3_corr << " Ha\n";
    std::cout << "Total corr:   " << ump3_result.e_corr_total << " Ha\n";
    std::cout << "UMP3 total:   " << ump3_result.e_total << " Ha\n";

    return 0;
}
