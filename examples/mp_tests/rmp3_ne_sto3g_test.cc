/**
 * @file rmp3_ne_sto3g_test.cc
 * @brief Fast RMP3 test on Ne atom with STO-3G basis (closed-shell)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/foundation/rmp2.h"
#include "mshqc/foundation/rmp3.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "====================================\n";
    std::cout << "  RMP3 Test: Ne/STO-3G\n";
    std::cout << "====================================\n";

    Molecule ne;
    ne.add_atom(10, 0.0, 0.0, 0.0); // Neon atom

    BasisSet basis("STO-3G", ne);
    std::cout << "Basis: STO-3G (" << basis.n_basis_functions() << " functions)\n";

    auto integrals = std::make_shared<IntegralEngine>(ne, basis);

    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    config.print_level = 0;

    RHF rhf(ne, basis, integrals, config);
    auto rhf_result = rhf.compute();

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "RHF energy:  " << rhf_result.energy_total << " Ha\n";

    foundation::RMP2 rmp2(rhf_result, basis, integrals);
    auto rmp2_result = rmp2.compute();
    std::cout << "MP2 corr:   " << rmp2_result.e_corr << " Ha\n";

    foundation::RMP3 rmp3(rhf_result, rmp2_result, basis, integrals);
    auto rmp3_result = rmp3.compute();

    std::cout << "MP3 corr:   " << rmp3_result.e_mp3 << " Ha\n";
    std::cout << "E(total):   " << rmp3_result.e_total << " Ha\n";

    return 0;
}
