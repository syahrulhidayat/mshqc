syahrul@syahrul:~/mshqc/include$ tree -I 'data' 'build' /home/syahrul/mshqc
build  [error opening dir]
/home/syahrul/mshqc
├── analisis grok.tex
├── cmake
│   ├── FindLibcint.cmake
│   ├── MSHQCConfig.cmake.in
│   └── scritpts
│       ├── check_installation.py
│       └── test_mshqc.py 
├── CMakeLists.txt
├── CONTRIBUTING.md 
├── docs
│   ├── debug analisis
│   │   ├── analisis grok.tex
│   │   ├── eri_transformer.cc
│   │   ├── eri_transformer.h
│   │   ├── eri_transformer.h.backup
│   │   ├── integrals.cc
│   │   ├── integrals.h
│   │   ├── scf.h
│   │   ├── uhf.cc
│   │   ├── ump2.cc
│   │   ├── ump2.h
│   │   ├── ump3.cc
│   │   ├── ump3.h
│   │   └── ump3_test.cc
│   ├── installation
│   │   ├── LINUX.md
│   │   ├── README.md
│   │   ├── TROUBLESHOOTING.md
│   │   └── WINDOWS.md
│   ├── QUICKSTART.md
│   └── TROUBLESHOOTING.md
├── examples
│   ├── ci_tests
│   │   ├── cisd_h2_test.cc
│   │   ├── cisd_onthefly_test.cc
│   │   ├── determinant_test.cc
│   │   ├── excitation_generator_test.cc
│   │   ├── h2_ci_matrix_element_test.cc
│   │   ├── h2_ci_sto3g_regression.cc
│   │   ├── h2_dissociation_ci_test.cc
│   │   ├── he_fci_regression.cc
│   │   ├── li_all_ci_methods_comparison.cc
│   │   ├── li_ccpvtz_all_methods_nist.cc
│   │   ├── li_ccpvtz_ci_benchmark.cc
│   │   ├── li_cisdt_benchmark.cc
│   │   ├── li_fci_validation_FIXED.cc
│   │   ├── li_uhf_cisd_test_FIXED.cc
│   │   ├── test_cholesky_li_sto3g.cc
│   │   ├── test_cholesky_mo_transform.cc
│   │   ├── test_cholesky_n10.cc
│   │   ├── test_cholesky_n15.cc
│   │   ├── test_cholesky_n20.cc
│   │   ├── test_cholesky_simple.cc
│   │   ├── test_cipsi_h2_sto3g.cc
│   │   ├── test_cipsi_li_ccpvtz.cc
│   │   ├── test_cipsi_simple.cc
│   │   ├── test_fci_vs_cipsi.cc
│   │   ├── test_integrated_mp_fci_driver.cc
│   │   ├── test_mp_fci_validation.cc
│   │   ├── test_mp_natural_orbitals_demo.cc
│   │   ├── test_natural_orbitals.cc
│   │   ├── timer.dat
│   │   └── validate_natural_orbital_logic.py
│   ├── dfmp2_integrals_test.cc
│   ├── dfmp2_test.cc
│   ├── gradient
│   │   ├── README.md
│   │   ├── test_gradient_h2.cc
│   │   ├── test_gradient_h2o_pvdz.cc
│   │   ├── test_gradient_li.cc
│   │   ├── test_gradient_lih.cc
│   │   └── test_optimize_h2.cc
│   ├── li_ccpvtz_test.cc
│   ├── mcscf
│   │   └── test_sacas_li.cc
│   ├── mcscf_tests
│   │   ├── casscf_h2_test.cc
│   │   ├── demo_natural_orbital_integration.cc
│   │   ├── h2_caspt2_test.cc
│   │   ├── h2_casscf_simple.cc
│   │   ├── he_caspt2_test.cc
│   │   ├── li_caspt2_test.cc
│   │   ├── li_caspt2_vs_mrmp2_ccpvtz.cc
│   │   ├── li_casscf_test.cc
│   │   ├── li_ci_integrals_debug.cc
│   │   ├── li_diagnostic.cc
│   │   ├── li_ground_state_fixed.cc
│   │   ├── li_uhf_caspt2_ccpvtz.cc
│   │   ├── test_li_ump3_ccpvtz.cc
│   │   └── test_li_ump3_sto3g.cc
│   ├── mp_tests
│   │   ├── cholesky_caspt2_test.cc
│   │   ├── df_caspt2_li_test.cc
│   │   ├── df_minimal_test.cc
│   │   ├── li_ccpvqz_hierarchy_test.cc
│   │   ├── mpn_hierarchy_complete.cc
│   │   ├── quick_mp_hierarchy_test.cc
│   │   ├── rmp2_test.cc
│   │   ├── rmp3_h2_ccpvdz_test.cc
│   │   ├── rmp3_h2o_ccpvdz_test.cc
│   │   ├── rmp3_h2o_sto3g_test.cc
│   │   ├── rmp3_h2_sto3g_test.cc
│   │   ├── rmp3_he2_test.cc
│   │   ├── rmp3_ne_sto3g_test.cc
│   │   ├── rmp3_test.cc
│   │   ├── t3_quick_test.cc
│   │   ├── test_3center_simple.cc
│   │   ├── test_eri3_availability.cc
│   │   ├── ump3_li_test.cc
│   │   ├── ump4_test.cc
│   │   ├── ump5_li_test.cc
│   │   └── wavefunction_test.cc
│   ├── rhf_he_test.cc
│   ├── rhf_test.cc
│   ├── rohf_test.cc
│   ├── test_new_ump3.cc
│   ├── uhf_test.cc
│   ├── ump2_test.cc
│   ├── ump3_pV5Z.cc
│   ├── ump3_pVTZ.cc
│   ├── ump3_test.cc
│   └── user_example
│       ├── CMakeLists.txt
│       └── main.cpp
├── include
│   └── mshqc
│       ├── basis.h
│       ├── ci
│       │   ├── cipsi.h
│       │   ├── cisd.h
│       │   ├── cisdt.h
│       │   ├── cis.h
│       │   ├── ci_utils.h
│       │   ├── davidson.h
│       │   ├── determinant.h
│       │   ├── excitation_generator.h
│       │   ├── external_space.h
│       │   ├── fci.h
│       │   ├── hamiltonian_sparse.h
│       │   ├── mrci.h
│       │   ├── natural_orbitals.h
│       │   ├── size_consistency.h
│       │   ├── slater_condon.h
│       │   ├── sparse_coo.h
│       │   ├── sparse_csr.h
│       │   ├── transition_density.h
│       │   └── wavefunction_analysis.h
│       ├── dfmp2.h
│       ├── foundation
│       │   ├── opdm.h
│       │   ├── rmp2.h
│       │   ├── rmp3.h
│       │   └── wavefunction.h
│       ├── gradient
│       │   ├── analytical_gradient.h
│       │   ├── gradient.h
│       │   └── optimizer.h
│       ├── integrals
│       │   ├── cholesky_eri.h
│       │   └── eri_transformer.h
│       ├── integrals.h
│       ├── integration
│       │   └── mp_ci_adapter.h
│       ├── libcint_wrapper.h
│       ├── mcscf
│       │   ├── active_space.h
│       │   ├── caspt2.h
│       │   ├── casscf.h
│       │   ├── cholesky_caspt2.h
│       │   ├── df_caspt2.h
│       │   ├── external_space.h
│       │   ├── mrmp2.h
│       │   ├── multi_root_ci.h
│       │   └── sa_casscf.h
│       ├── molecule.h
│       ├── mp
│       │   ├── mp_density.h
│       │   ├── ump4.h
│       │   └── ump5.h
│       ├── mp2.h
│       ├── mpn_hierarchy.h
│       ├── omp3.h
│       ├── scf.h
│       ├── spherical_integration.h
│       ├── spherical_transformer.h
│       ├── ump2.h
│       ├── ump2.h.backup
│       ├── ump2.h.backup2
│       ├── ump3.h
│       ├── ump3.h.backupp
│       └── validation
│           └── mp_vs_fci.h
├── install_dependencies.sh
├── install.sh
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── python
│   ├── bindings.cc
│   ├── __init__.py
│   ├── mshqc
│   │   └── __init__.py
│   └── mshqc.egg-info
│       ├── dependency_links.txt
│       ├── not-zip-safe
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
├── README.md
├── README_old.md
├── requirements.txt
├── setup.py
├── src
│   ├── ci
│   │   ├── cipsi.cc
│   │   ├── cis.cc
│   │   ├── cisd.cc
│   │   ├── cisdt.cc
│   │   ├── davidson.cc
│   │   ├── determinant.cc
│   │   ├── excitation_generator.cc
│   │   ├── fci.cc
│   │   ├── hamiltonian_sparse.cc
│   │   ├── hamiltonian_sparse.cc.backup
│   │   ├── mrci.cc
│   │   ├── natural_orbitals.cc
│   │   ├── size_consistency.cc
│   │   ├── slater_condon.cc
│   │   ├── sparse_coo.cc
│   │   ├── transition_density.cc
│   │   └── wavefunction_analysis.cc
│   ├── core
│   │   ├── basis.cc
│   │   ├── integrals.cc
│   │   ├── libcint_wrapper.cc
│   │   ├── molecule.cc
│   │   ├── spherical_integration.cc
│   │   └── spherical_transformer.cc
│   ├── foundation
│   │   ├── opdm.cc
│   │   ├── rmp2.cc
│   │   ├── rmp3.cc
│   │   └── wavefunction.cc
│   ├── gradient
│   │   ├── analytical_gradient.cc
│   │   ├── numerical_gradient.cc
│   │   └── optimizer.cc
│   ├── integrals
│   │   ├── cholesky_eri.cc
│   │   └── eri_transformer.cc
│   ├── integration
│   │   └── mp_ci_adapter.cc
│   ├── mcscf
│   │   ├── active_space.cc
│   │   ├── caspt2.cc
│   │   ├── casscf.cc
│   │   ├── cholesky_caspt2.cc
│   │   ├── df_caspt2.cc
│   │   ├── external_space.cc
│   │   ├── mrmp2.cc
│   │   ├── multi_root_ci.cc
│   │   └── sa_casscf.cc
│   ├── mp
│   │   ├── mp_density.cc
│   │   ├── ump4.cc
│   │   └── ump5.cc
│   ├── mp2
│   │   ├── dfmp2.cc
│   │   ├── omp2.cc
│   │   ├── semicanonical.cc
│   │   ├── ump2.cc
│   │   └── ump2.cc.backup
│   ├── mp3
│   │   ├── omp3.cc
│   │   ├── ump3_backup.cc.backup
│   │   ├── ump3.cc
│   │   ├── ump3.cc.backup
│   │   ├── ump3_debug.cc
│   │   └── ump3_energy.cc.new
│   ├── scf
│   │   ├── diis.cc
│   │   ├── rhf.cc
│   │   ├── rohf.cc
│   │   ├── uhf.cc
│   │   └── uhf.cc.backup
│   └── validation
│       └── mp_vs_fci.cc
├── struktru.md
├── test2.py
├── test.py
├── tests
│   ├── li_ccpvtz_psi4.py
│   ├── li_omp2_psi4.py
│   ├── test_ump3_complete_li_sto3g.cc
│   ├── test_ump3_oxygen.cc
│   └── timer.dat
├── ump3_debug_patch.cc
├── ump3_test
└── ump3_test_fixed

40 directories, 261 files
syahrul@syahrul:~/mshqc/include$ 

