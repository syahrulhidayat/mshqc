#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Include MSH-QC headers
#include <mshqc/molecule.h>
#include <mshqc/basis.h>
#include <mshqc/scf.h>
#include <mshqc/integrals.h>

// MP2/MP3
#include <mshqc/mp2.h>
#include <mshqc/ump2.h>
#include <mshqc/ump3.h>
#include <mshqc/omp3.h>
#include <mshqc/dfmp2.h>

// MP4/MP5
#include <mshqc/mp/ump4.h>
#include <mshqc/mp/ump5.h>

// Foundation
#include <mshqc/foundation/rmp2.h>
#include <mshqc/foundation/rmp3.h>
#include <mshqc/foundation/wavefunction.h>
#include <mshqc/foundation/opdm.h>

// CI methods
#include <mshqc/ci/determinant.h>
#include <mshqc/ci/cis.h>
#include <mshqc/ci/cisd.h>
#include <mshqc/ci/cisdt.h>
#include <mshqc/ci/fci.h>
#include <mshqc/ci/mrci.h>
#include <mshqc/ci/cipsi.h>
#include <mshqc/ci/davidson.h>
#include <mshqc/ci/natural_orbitals.h>
#include <mshqc/ci/wavefunction_analysis.h>

// MCSCF
#include <mshqc/mcscf/active_space.h>
#include <mshqc/mcscf/casscf.h>
#include <mshqc/mcscf/caspt2.h>
#include <mshqc/mcscf/df_caspt2.h>
#include <mshqc/mcscf/mrmp2.h>
#include <mshqc/mcscf/sa_casscf.h>

// Gradient
#include <mshqc/gradient/gradient.h>
#include <mshqc/gradient/optimizer.h>

namespace py = pybind11;
using namespace mshqc;

// Helper to convert Eigen matrices to numpy arrays
py::array_t<double> eigen_to_numpy(const Eigen::MatrixXd& mat) {
    return py::array_t<double>(
        {mat.rows(), mat.cols()},
        {sizeof(double) * mat.cols(), sizeof(double)},
        mat.data()
    );
}

py::array_t<double> eigen_to_numpy_vec(const Eigen::VectorXd& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "MSH-QC: Comprehensive Quantum Mechanics Library for Computational Chemistry";

    // ========================================================================
    // CORE MODULE: Molecule, Basis Set, Integrals
    // ========================================================================
    
    py::class_<Molecule>(m, "Molecule", "Molecular system specification")
        .def(py::init<>())
        .def("add_atom", &Molecule::add_atom,
             py::arg("atomic_number"), py::arg("x"), py::arg("y"), py::arg("z"),
             "Add an atom to the molecule")
        .def("natoms", &Molecule::natoms, "Number of atoms")
        .def("nelectrons", &Molecule::nelectrons, "Number of electrons")
        .def("charge", &Molecule::charge, "Molecular charge")
        .def("multiplicity", &Molecule::multiplicity, "Spin multiplicity")
        .def("nuclear_repulsion", &Molecule::nuclear_repulsion, "Nuclear repulsion energy");

    py::class_<BasisSet>(m, "BasisSet", "Gaussian basis set")
        .def(py::init<>())
        .def("load", &BasisSet::load,
             py::arg("basis_name"), py::arg("molecule"),
             "Load basis set for molecule")
        .def("nbasis", &BasisSet::nbasis, "Number of basis functions")
        .def("nbf", &BasisSet::nbf, "Number of basis functions (alias)");

    py::class_<IntegralEngine>(m, "IntegralEngine", "Compute electron integrals")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("compute_overlap", &IntegralEngine::compute_overlap,
             "Compute overlap matrix")
        .def("compute_kinetic", &IntegralEngine::compute_kinetic,
             "Compute kinetic energy matrix")
        .def("compute_nuclear", &IntegralEngine::compute_nuclear,
             "Compute nuclear attraction matrix")
        .def("compute_eri", &IntegralEngine::compute_eri,
             "Compute electron repulsion integrals");

    // ========================================================================
    // SCF MODULE: Hartree-Fock methods
    // ========================================================================
    
    py::class_<SCFResult>(m, "SCFResult", "Self-consistent field calculation result")
        .def(py::init<>())
        .def_readwrite("energy", &SCFResult::energy, "Total energy")
        .def_readwrite("converged", &SCFResult::converged, "Convergence status")
        .def_readwrite("n_occ", &SCFResult::n_occ, "Number of occupied orbitals")
        .def_readwrite("n_vir", &SCFResult::n_vir, "Number of virtual orbitals")
        .def_property("C",
            [](const SCFResult& r) { return eigen_to_numpy(r.C); },
            nullptr,
            "MO coefficients")
        .def_property("e",
            [](const SCFResult& r) { return eigen_to_numpy_vec(r.e); },
            nullptr,
            "Orbital energies")
        .def_property("D",
            [](const SCFResult& r) { return eigen_to_numpy(r.D); },
            nullptr,
            "Density matrix");

    py::class_<RHF>(m, "RHF", "Restricted Hartree-Fock")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("solve", &RHF::solve, "Perform RHF calculation")
        .def("get_energy", &RHF::get_energy, "Get current energy");

    py::class_<ROHF>(m, "ROHF", "Restricted Open-shell Hartree-Fock")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("solve", &ROHF::solve, "Perform ROHF calculation")
        .def("get_energy", &ROHF::get_energy, "Get current energy");

    py::class_<UHF>(m, "UHF", "Unrestricted Hartree-Fock")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("solve", &UHF::solve, "Perform UHF calculation")
        .def("get_energy", &UHF::get_energy, "Get current energy")
        .def("get_spin_contamination", &UHF::get_spin_contamination,
             "Spin contamination <S^2>");

    // ========================================================================
    // MP2/MP3 MODULE: Møller-Plesset Perturbation Theory
    // ========================================================================
    
    py::class_<MP2Result>(m, "MP2Result", "MP2 calculation result")
        .def(py::init<>())
        .def_readwrite("e_scf", &MP2Result::e_scf, "SCF reference energy")
        .def_readwrite("e_corr", &MP2Result::e_corr, "MP2 correlation energy")
        .def_readwrite("e_total", &MP2Result::e_total, "Total MP2 energy");

    py::class_<RMP2>(m, "RMP2", "Restricted MP2")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &RMP2::solve, "Compute RMP2 energy")
        .def("get_energy", &RMP2::get_energy, "Get MP2 energy");

    py::class_<UMP2>(m, "UMP2", "Unrestricted MP2")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &UMP2::solve, "Compute UMP2 energy")
        .def("get_energy", &UMP2::get_energy, "Get MP2 energy");

    py::class_<DFMP2>(m, "DFMP2", "Density-Fitting MP2")
        .def(py::init<const SCFResult&, const BasisSet&>())
        .def("solve", &DFMP2::solve, "Compute DF-MP2 energy")
        .def("get_energy", &DFMP2::get_energy, "Get DF-MP2 energy");

    py::class_<UMP3Result>(m, "UMP3Result", "UMP3 calculation result")
        .def(py::init<>())
        .def_readwrite("e_scf", &UMP3Result::e_scf, "SCF reference energy")
        .def_readwrite("e_mp2", &UMP3Result::e_mp2, "MP2 correlation energy")
        .def_readwrite("e_mp3", &UMP3Result::e_mp3, "MP3 correlation energy")
        .def_readwrite("e_total", &UMP3Result::e_total, "Total MP3 energy");

    py::class_<UMP3>(m, "UMP3", "Unrestricted MP3")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &UMP3::solve, "Compute UMP3 energy")
        .def("get_energy", &UMP3::get_energy, "Get MP3 energy");

    py::class_<RMP3>(m, "RMP3", "Restricted MP3")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &RMP3::solve, "Compute RMP3 energy")
        .def("get_energy", &RMP3::get_energy, "Get MP3 energy");

    // UMP4/UMP5
    py::class_<UMP4>(m, "UMP4", "Unrestricted MP4")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &UMP4::solve, "Compute UMP4 energy")
        .def("get_energy", &UMP4::get_energy, "Get MP4 energy");

    py::class_<UMP5>(m, "UMP5", "Unrestricted MP5")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &UMP5::solve, "Compute UMP5 energy")
        .def("get_energy", &UMP5::get_energy, "Get MP5 energy");

    // ========================================================================
    // CI MODULE: Configuration Interaction
    // ========================================================================
    
    py::class_<Determinant>(m, "Determinant", "Slater determinant")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def("excite", &Determinant::excite, "Single excitation")
        .def("phase", &Determinant::phase, "Sign of excitation");

    py::class_<CIResult>(m, "CIResult", "Configuration interaction result")
        .def(py::init<>())
        .def_readwrite("energy", &CIResult::energy, "CI energy")
        .def_readwrite("e_corr", &CIResult::e_corr, "Correlation energy")
        .def_property("coefficients",
            [](const CIResult& r) { return eigen_to_numpy_vec(r.coefficients); },
            nullptr,
            "CI coefficients");

    py::class_<CIS>(m, "CIS", "Configuration Interaction Singles")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &CIS::solve, "Perform CIS calculation")
        .def("get_energy", &CIS::get_energy, "Get CIS energy")
        .def("get_excitation_energy", &CIS::get_excitation_energy,
             py::arg("state"), "Get excitation energy for state");

    py::class_<CISD>(m, "CISD", "Configuration Interaction Singles and Doubles")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &CISD::solve, "Perform CISD calculation")
        .def("get_energy", &CISD::get_energy, "Get CISD energy");

    py::class_<CISDT>(m, "CISDT", "CI Singles, Doubles, and Triples")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &CISDT::solve, "Perform CISDT calculation")
        .def("get_energy", &CISDT::get_energy, "Get CISDT energy");

    py::class_<FCI>(m, "FCI", "Full Configuration Interaction")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &FCI::solve, "Perform FCI calculation")
        .def("get_energy", &FCI::get_energy, "Get FCI energy");

    py::class_<MRCI>(m, "MRCI", "Multi-Reference Configuration Interaction")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("set_active_space", &MRCI::set_active_space,
             py::arg("n_electrons"), py::arg("n_orbitals"))
        .def("solve", &MRCI::solve, "Perform MRCI calculation")
        .def("get_energy", &MRCI::get_energy, "Get MRCI energy");

    py::class_<CIPSI>(m, "CIPSI", "Configuration Interaction using Perturbative Selection Iteratively")
        .def(py::init<const SCFResult&, const IntegralEngine&>())
        .def("solve", &CIPSI::solve,
             py::arg("epsilon_var") = 1e-6,
             "Perform CIPSI calculation with selection threshold")
        .def("get_energy", &CIPSI::get_energy, "Get CIPSI energy")
        .def("get_variational_energy", &CIPSI::get_variational_energy,
             "Get variational energy")
        .def("get_pt2_correction", &CIPSI::get_pt2_correction,
             "Get PT2 correction");

    // ========================================================================
    // MCSCF MODULE: Multi-Configurational SCF
    // ========================================================================
    
    py::class_<ActiveSpace>(m, "ActiveSpace", "CAS active space definition")
        .def(py::init<int, int>(),
             py::arg("n_electrons"), py::arg("n_orbitals"))
        .def("n_electrons", &ActiveSpace::n_electrons, "Active electrons")
        .def("n_orbitals", &ActiveSpace::n_orbitals, "Active orbitals");

    py::class_<CASSCFResult>(m, "CASSCFResult", "CASSCF result")
        .def(py::init<>())
        .def_readwrite("energy", &CASSCFResult::energy, "CASSCF energy")
        .def_readwrite("converged", &CASSCFResult::converged, "Convergence status");

    py::class_<CASSCF>(m, "CASSCF", "Complete Active Space SCF")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("set_active_space", &CASSCF::set_active_space,
             py::arg("n_electrons"), py::arg("n_orbitals"),
             "Set CAS active space")
        .def("solve", &CASSCF::solve, "Perform CASSCF calculation")
        .def("get_energy", &CASSCF::get_energy, "Get CASSCF energy");

    py::class_<SACASSCF>(m, "SACASSCF", "State-Averaged CASSCF")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("set_active_space", &SACASSCF::set_active_space,
             py::arg("n_electrons"), py::arg("n_orbitals"))
        .def("set_n_states", &SACASSCF::set_n_states,
             py::arg("n_states"), "Set number of states to average")
        .def("solve", &SACASSCF::solve, "Perform SA-CASSCF calculation")
        .def("get_energy", &SACASSCF::get_energy,
             py::arg("state"), "Get energy for specific state");

    py::class_<CASPT2>(m, "CASPT2", "Complete Active Space Perturbation Theory")
        .def(py::init<const CASSCFResult&, const IntegralEngine&>())
        .def("solve", &CASPT2::solve, "Perform CASPT2 calculation")
        .def("get_energy", &CASPT2::get_energy, "Get CASPT2 energy")
        .def("get_correlation_energy", &CASPT2::get_correlation_energy,
             "Get CASPT2 correlation energy");

    py::class_<DFCASPT2>(m, "DFCASPT2", "Density-Fitting CASPT2")
        .def(py::init<const CASSCFResult&, const BasisSet&>())
        .def("solve", &DFCASPT2::solve, "Perform DF-CASPT2 calculation")
        .def("get_energy", &DFCASPT2::get_energy, "Get DF-CASPT2 energy");

    py::class_<MRMP2>(m, "MRMP2", "Multi-Reference MP2")
        .def(py::init<const CASSCFResult&, const IntegralEngine&>())
        .def("solve", &MRMP2::solve, "Perform MRMP2 calculation")
        .def("get_energy", &MRMP2::get_energy, "Get MRMP2 energy");

    // ========================================================================
    // GRADIENT MODULE: Geometry Optimization
    // ========================================================================
    
    py::class_<NumericalGradient>(m, "NumericalGradient", "Numerical gradient calculator")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("compute_rhf_gradient", &NumericalGradient::compute_rhf_gradient,
             "Compute RHF gradient")
        .def("compute_mp2_gradient", &NumericalGradient::compute_mp2_gradient,
             "Compute MP2 gradient");

    py::class_<GeometryOptimizer>(m, "GeometryOptimizer", "Molecular geometry optimizer")
        .def(py::init<const Molecule&, const BasisSet&>())
        .def("set_method", &GeometryOptimizer::set_method,
             py::arg("method"), "Set optimization method (SD, CG, BFGS)")
        .def("optimize", &GeometryOptimizer::optimize,
             "Optimize molecular geometry")
        .def("get_optimized_energy", &GeometryOptimizer::get_optimized_energy,
             "Get optimized energy")
        .def("get_optimized_geometry", &GeometryOptimizer::get_optimized_geometry,
             "Get optimized geometry");

    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================
    
    m.def("create_h2_molecule", [](double distance = 0.74) {
        Molecule mol;
        mol.add_atom(1, 0.0, 0.0, -distance/2.0);  // H
        mol.add_atom(1, 0.0, 0.0, distance/2.0);   // H
        return mol;
    }, py::arg("distance") = 0.74, "Create H2 molecule");

    m.def("create_water_molecule", []() {
        Molecule mol;
        mol.add_atom(8, 0.0, 0.0, 0.119262);    // O
        mol.add_atom(1, 0.0, 0.757, -0.477023);  // H
        mol.add_atom(1, 0.0, -0.757, -0.477023); // H
        return mol;
    }, "Create water molecule");

    m.def("create_li_atom", []() {
        Molecule mol;
        mol.add_atom(3, 0.0, 0.0, 0.0);  // Li
        return mol;
    }, "Create lithium atom");

    m.def("create_he_atom", []() {
        Molecule mol;
        mol.add_atom(2, 0.0, 0.0, 0.0);  // He
        return mol;
    }, "Create helium atom");

    m.def("create_ne_atom", []() {
        Molecule mol;
        mol.add_atom(10, 0.0, 0.0, 0.0);  // Ne
        return mol;
    }, "Create neon atom");

    // Version info
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "0.1.0";
#endif
}