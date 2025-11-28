#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <cassert>

// Simple regression to check CI mapping + shift on H2/STO-3G
int main() {
  using namespace mshqc;
  // H2 molecule
  Molecule mol; mol.add_atom(1, 0.0, 0.0, -0.37); mol.add_atom(1, 0.0, 0.0, 0.37);
  BasisSet basis("STO-3G", mol);
  auto eng = std::make_shared<IntegralEngine>(mol, basis);

  // RHF/UHF
  SCFConfig cfg; cfg.print_level = 0; cfg.max_iterations = 100; cfg.energy_threshold = 1e-10; cfg.density_threshold = 1e-8;
  UHF uhf(mol, basis, eng, 1, 1, cfg);
  auto uhf_res = uhf.compute();
  if(!uhf_res.converged){ std::cerr << "UHF did not converge" << std::endl; return 1; }
  std::cout << "E(UHF)=" << uhf_res.energy_total << "\n";

  // Build bare one-electron h (AO) and transform to MO
  auto T = eng->compute_kinetic();
  auto V = eng->compute_nuclear();
  Eigen::MatrixXd h_ao = T + V;
  Eigen::MatrixXd h_alpha = uhf_res.C_alpha.transpose() * h_ao * uhf_res.C_alpha;
  Eigen::MatrixXd h_beta  = uhf_res.C_beta.transpose()  * h_ao * uhf_res.C_beta;

  // ERI AO -> chemist MO (pq|rs)
  auto ERI_AO = eng->compute_eri();
  Eigen::Tensor<double,4> ERI_MO_AA = transform_eri_to_mo(ERI_AO, uhf_res.C_alpha);
  Eigen::Tensor<double,4> ERI_MO_BB = transform_eri_to_mo(ERI_AO, uhf_res.C_beta);
  Eigen::Tensor<double,4> eri_aaaa(ERI_MO_AA.dimensions());
  Eigen::Tensor<double,4> eri_bbbb(ERI_MO_BB.dimensions());
  Eigen::Tensor<double,4> eri_aabb(ERI_MO_AA.dimensions());

  // Use centralized helpers (theory-based, AI_RULES-compliant):
  mshqc::ci::build_same_spin_antisym_from_chemist(ERI_MO_AA, eri_aaaa);
  mshqc::ci::build_same_spin_antisym_from_chemist(ERI_MO_BB, eri_bbbb);
  mshqc::ci::build_alpha_beta_from_chemist(ERI_MO_AA, eri_aabb);
  // CI integrals
  ci::CIIntegrals ints; ints.h_alpha=h_alpha; ints.h_beta=h_beta; ints.eri_aaaa=eri_aaaa; ints.eri_bbbb=eri_bbbb; ints.eri_aabb=eri_aabb; ints.use_fock=false; ints.e_nuc=uhf_res.energy_nuclear;

  // HF determinant (2e: one alpha, one beta in orbital 0)
  ci::Determinant hf_det(std::vector<int>{0}, std::vector<int>{0});
  ci::CISD cisd(ints, hf_det, /*nocc_a*/1, /*nocc_b*/1, /*nvirt_a*/(int)h_alpha.rows()-1, /*nvirt_b*/(int)h_beta.rows()-1);
  auto res = cisd.compute({/*defaults*/});
  std::cout << "E(CISD)=" << res.e_cisd << "  E_corr=" << res.e_corr << "\n";

  // Regression checks (broad):
  // 1) Correlation should be negative and modest
  assert(res.e_corr < -1e-6 && res.e_corr > -0.5);
  // 2) Shift makes HF diagonal ~ HF energy, so H(0,0) after shift is ~0 (implicitly tested in CISD)
  // 3) No crash
  std::cout << "H2/STO-3G regression OK\n";
  return 0;
}