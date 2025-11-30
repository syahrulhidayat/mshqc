// python/bindings.cc - Complete Python bindings for MSHQC library
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

// Core headers
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"

// MP2/MP3 headers
#include "mshqc/mp2.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include "mshqc/dfmp2.h"
#include "mshqc/foundation/rmp2.h"
#include "mshqc/foundation/rmp3.h"
#include "mshqc/foundation/wavefunction.h"

// CI headers
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/cis.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/cisdt.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/cipsi.h"

// MCSCF headers
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/caspt2.h"

// Gradient headers
#include "mshqc/gradient/gradient.h"
#include "mshqc/gradient/optimizer.h"

namespace py = pybind11;
using namespace mshqc;

PYBIND11_MODULE(mshqc, m) {
    m.doc() = "MSHQC: Modern Quantum Chemistry Library";

    // ========================================================================
    // Core Classes: Molecule, Basis, Integrals
    // ========================================================================
    
    py::class_<Atom>(m, "Atom")
        .def(py::init<int, double, double, double>(),
             py::arg("atomic_number"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("atomic_number", &Atom::atomic_number)
        .def_readwrite("x", &Atom::x)
        .def_readwrite("y", &Atom::y)
        .def_readwrite("z", &Atom::z)
        .def("position", &Atom::position);

    py::class_<Molecule>(m, "Molecule")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("charge"), py::arg("multiplicity"))
        .def("add_atom", py::overload_cast<int, double, double, double>(&Molecule::add_atom),
             py::arg("Z"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("add_atom", py::overload_cast<const Atom&>(&Molecule::add_atom),
             py::arg("atom"))
        .def("n_atoms", &Molecule::n_atoms)
        .def("atom", &Molecule::atom, py::arg("i"))
        .def("atoms", &Molecule::atoms)
        .def("total_nuclear_charge", &Molecule::total_nuclear_charge)
        .def("n_electrons", &Molecule::n_electrons)
        .def("charge", &Molecule::charge)
        .def("set_charge", &Molecule::set_charge, py::arg("q"))
        .def("multiplicity", &Molecule::multiplicity)
        .def("set_multiplicity", &Molecule::set_multiplicity, py::arg("m"))
        .def("nuclear_repulsion_energy", &Molecule::nuclear_repulsion_energy);

    py::enum_<AngularMomentum>(m, "AngularMomentum")
        .value("S", AngularMomentum::S)
        .value("P", AngularMomentum::P)
        .value("D", AngularMomentum::D)
        .value("F", AngularMomentum::F)
        .value("G", AngularMomentum::G)
        .value("H", AngularMomentum::H);

    py::class_<GaussianPrimitive>(m, "GaussianPrimitive")
        .def(py::init<double, double>(), py::arg("exponent"), py::arg("coefficient"))
        .def_readwrite("exponent", &GaussianPrimitive::exponent)
        .def_readwrite("coefficient", &GaussianPrimitive::coefficient);

    py::class_<Shell>(m, "Shell")
        .def(py::init<>())
        .def_readwrite("angular_momentum", &Shell::angular_momentum)
        .def_readwrite("center", &Shell::center)
        .def_readwrite("primitives", &Shell::primitives);

    py::class_<BasisSet>(m, "BasisSet")
        .def(py::init<>())
        .def(py::init<const std::string&, const Molecule&, const std::string&>(),
             py::arg("basis_name"), py::arg("mol"), py::arg("basis_dir") = "../data/basis")
        .def("read_gbs", &BasisSet::read_gbs, py::arg("basis_file"), py::arg("mol"))
        .def("add_shell", &BasisSet::add_shell, py::arg("shell"))
        .def("n_shells", &BasisSet::n_shells)
        .def("n_basis_functions", &BasisSet::n_basis_functions)
        .def("shell", &BasisSet::shell, py::arg("i"))
        .def("name", &BasisSet::name);

    py::class_<IntegralEngine, std::shared_ptr<IntegralEngine>>(m, "IntegralEngine")
        .def(py::init<const Molecule&, const BasisSet&>(),
             py::arg("mol"), py::arg("basis"))
        .def("compute_overlap", &IntegralEngine::compute_overlap)
        .def("compute_kinetic", &IntegralEngine::compute_kinetic)
        .def("compute_nuclear", &IntegralEngine::compute_nuclear)
        .def("compute_eri", &IntegralEngine::compute_eri);

    // ========================================================================
    // SCF: Configuration, Results, and Solvers
    // ========================================================================
    
    py::class_<SCFConfig>(m, "SCFConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &SCFConfig::max_iterations)
        .def_readwrite("energy_threshold", &SCFConfig::energy_threshold)
        .def_readwrite("density_threshold", &SCFConfig::density_threshold)
        .def_readwrite("diis_threshold", &SCFConfig::diis_threshold)
        .def_readwrite("diis_max_vectors", &SCFConfig::diis_max_vectors)
        .def_readwrite("print_level", &SCFConfig::print_level)
        .def_readwrite("level_shift", &SCFConfig::level_shift);

    py::class_<SCFResult>(m, "SCFResult")
        .def(py::init<>())
        .def_readwrite("energy_electronic", &SCFResult::energy_electronic)
        .def_readwrite("energy_nuclear", &SCFResult::energy_nuclear)
        .def_readwrite("energy_total", &SCFResult::energy_total)
        .def_readwrite("orbital_energies_alpha", &SCFResult::orbital_energies_alpha)
        .def_readwrite("orbital_energies_beta", &SCFResult::orbital_energies_beta)
        .def_readwrite("C_alpha", &SCFResult::C_alpha)
        .def_readwrite("C_beta", &SCFResult::C_beta)
        .def_readwrite("P_alpha", &SCFResult::P_alpha)
        .def_readwrite("P_beta", &SCFResult::P_beta)
        .def_readwrite("F_alpha", &SCFResult::F_alpha)
        .def_readwrite("F_beta", &SCFResult::F_beta)
        .def_readwrite("iterations", &SCFResult::iterations)
        .def_readwrite("converged", &SCFResult::converged)
        .def_readwrite("gradient_norm", &SCFResult::gradient_norm)
        .def_readwrite("n_occ_alpha", &SCFResult::n_occ_alpha)
        .def_readwrite("n_occ_beta", &SCFResult::n_occ_beta);

    py::class_<RHF>(m, "RHF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const SCFConfig&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), 
             py::arg("config") = SCFConfig())
        .def("compute", &RHF::compute)
        .def("energy", &RHF::energy)
        .def("nbasis", &RHF::nbasis)
        .def("n_occ", &RHF::n_occ);

    py::class_<UHF>(m, "UHF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const SCFConfig&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), 
             py::arg("config") = SCFConfig())
        .def("compute", &UHF::compute)
        .def("energy", &UHF::energy);

    py::class_<ROHF>(m, "ROHF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const SCFConfig&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), 
             py::arg("config") = SCFConfig())
        .def("compute", &ROHF::compute)
        .def("energy", &ROHF::energy);

    // ========================================================================
    // MP2/MP3 Methods
    // ========================================================================
    
    py::class_<MP2Result>(m, "MP2Result")
        .def(py::init<>())
        .def_readwrite("energy_scf", &MP2Result::energy_scf)
        .def_readwrite("energy_mp2_ss", &MP2Result::energy_mp2_ss)
        .def_readwrite("energy_mp2_os", &MP2Result::energy_mp2_os)
        .def_readwrite("energy_mp2_corr", &MP2Result::energy_mp2_corr)
        .def_readwrite("energy_total", &MP2Result::energy_total)
        .def_readwrite("n_occ_alpha", &MP2Result::n_occ_alpha)
        .def_readwrite("n_occ_beta", &MP2Result::n_occ_beta)
        .def_readwrite("n_virt_alpha", &MP2Result::n_virt_alpha)
        .def_readwrite("n_virt_beta", &MP2Result::n_virt_beta);

    py::class_<UMP2Result>(m, "UMP2Result")
        .def(py::init<>())
        .def_readwrite("e_corr_ss_aa", &UMP2Result::e_corr_ss_aa)
        .def_readwrite("e_corr_ss_bb", &UMP2Result::e_corr_ss_bb)
        .def_readwrite("e_corr_os", &UMP2Result::e_corr_os)
        .def_readwrite("e_corr_total", &UMP2Result::e_corr_total)
        .def_readwrite("e_total", &UMP2Result::e_total);

    py::class_<RMP2Result>(m, "RMP2Result")
        .def(py::init<>())
        .def_readwrite("e_scf", &RMP2Result::e_scf)
        .def_readwrite("e_corr", &RMP2Result::e_corr)
        .def_readwrite("e_total", &RMP2Result::e_total);

    py::class_<RMP3Result>(m, "RMP3Result")
        .def(py::init<>())
        .def_readwrite("e_scf", &RMP3Result::e_scf)
        .def_readwrite("e_mp2", &RMP3Result::e_mp2)
        .def_readwrite("e_mp3", &RMP3Result::e_mp3)
        .def_readwrite("e_total", &RMP3Result::e_total);

    // MP2 classes
    py::class_<ROMP2>(m, "ROMP2")
        .def(py::init<const SCFResult&, std::shared_ptr<IntegralEngine>>(),
             py::arg("scf_result"), py::arg("integrals"))
        .def("compute", &ROMP2::compute);

    py::class_<UMP2>(m, "UMP2")
        .def(py::init<const SCFResult&, std::shared_ptr<IntegralEngine>>(),
             py::arg("uhf_result"), py::arg("integrals"))
        .def("compute", &UMP2::compute);

    // Note: DFMP2 constructor needs SCFResult, BasisSet, AND IntegralEngine based on header
    py::class_<DFMP2>(m, "DFMP2")
        .def(py::init<const SCFResult&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("scf_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &DFMP2::compute);

    py::class_<RMP2>(m, "RMP2")
        .def(py::init<const SCFResult&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("rhf_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &RMP2::compute);

    py::class_<RMP3>(m, "RMP3")
        .def(py::init<const SCFResult&, const RMP2Result&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("rhf_result"), py::arg("rmp2_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &RMP3::compute);

    py::class_<UMP3>(m, "UMP3")
        .def(py::init<const SCFResult&, std::shared_ptr<IntegralEngine>>(),
             py::arg("uhf_result"), py::arg("integrals"))
        .def("compute", &UMP3::compute);

    // ========================================================================
    // CI Methods
    // ========================================================================
    
    py::class_<ci::Determinant>(m, "Determinant")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&>(),
             py::arg("alpha_occ"), py::arg("beta_occ"))
        .def("n_alpha", &ci::Determinant::n_alpha)
        .def("n_beta", &ci::Determinant::n_beta)
        .def("alpha_occ", &ci::Determinant::alpha_occ)
        .def("beta_occ", &ci::Determinant::beta_occ)
        .def("is_alpha_occupied", &ci::Determinant::is_alpha_occupied, py::arg("i"))
        .def("is_beta_occupied", &ci::Determinant::is_beta_occupied, py::arg("i"))
        .def("excitation_level", &ci::Determinant::excitation_level, py::arg("other"));

    py::class_<ci::CIIntegrals>(m, "CIIntegrals")
        .def(py::init<>())
        .def_readwrite("h_mo", &ci::CIIntegrals::h_mo)
        .def_readwrite("eri_mo", &ci::CIIntegrals::eri_mo)
        .def_readwrite("e_nuc", &ci::CIIntegrals::e_nuc);

    py::class_<ci::CIResult>(m, "CIResult")
        .def(py::init<>())
        .def_readwrite("energies", &ci::CIResult::energies)
        .def_readwrite("ci_vectors", &ci::CIResult::ci_vectors)
        .def_readwrite("determinants", &ci::CIResult::determinants)
        .def_readwrite("converged", &ci::CIResult::converged)
        .def_readwrite("n_iterations", &ci::CIResult::n_iterations);

    py::class_<ci::CIS>(m, "CIS")
        .def(py::init<const ci::CIIntegrals&, const ci::Determinant&, int, int, int, int>(),
             py::arg("ints"), py::arg("hf_det"), 
             py::arg("n_occ_alpha"), py::arg("n_occ_beta"),
             py::arg("n_virt_alpha"), py::arg("n_virt_beta"))
        .def("compute", &ci::CIS::compute, py::arg("n_roots") = 5);

    py::class_<ci::CISD>(m, "CISD")
        .def(py::init<const ci::CIIntegrals&, const ci::Determinant&, int, int, int, int>(),
             py::arg("ints"), py::arg("hf_det"),
             py::arg("n_occ_alpha"), py::arg("n_occ_beta"),
             py::arg("n_virt_alpha"), py::arg("n_virt_beta"))
        .def("compute", &ci::CISD::compute);

    py::class_<ci::CISDT>(m, "CISDT")
        .def(py::init<const ci::CIIntegrals&, const ci::Determinant&, int, int, int, int>(),
             py::arg("ints"), py::arg("hf_det"),
             py::arg("n_occ_alpha"), py::arg("n_occ_beta"),
             py::arg("n_virt_alpha"), py::arg("n_virt_beta"))
        .def("compute", &ci::CISDT::compute);

    py::class_<ci::FCI>(m, "FCI")
        .def(py::init<const ci::CIIntegrals&, int, int, int, int>(),
             py::arg("ints"), py::arg("n_orbitals"),
             py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_roots") = 1)
        .def("compute", &ci::FCI::compute);

    py::class_<ci::CIPSI>(m, "CIPSI")
        .def(py::init<const ci::CIIntegrals&, int, int, int>(),
             py::arg("ints"), py::arg("n_orbitals"),
             py::arg("n_alpha"), py::arg("n_beta"))
        .def("compute", &ci::CIPSI::compute,
             py::arg("energy_threshold") = 1e-6,
             py::arg("max_iterations") = 20);

    // ========================================================================
    // MCSCF Methods
    // ========================================================================
    
    py::class_<ActiveSpace>(m, "ActiveSpace")
        .def(py::init<>())
        .def(py::init<int, int, int, int>(),
             py::arg("n_inactive"), py::arg("n_active"), 
             py::arg("n_virtual"), py::arg("n_elec_active"))
        .def_static("CAS", &ActiveSpace::CAS,
                   py::arg("n_elec"), py::arg("n_orb"),
                   py::arg("n_total_orb"), py::arg("n_total_elec"))
        .def("n_inactive", &ActiveSpace::n_inactive)
        .def("n_active", &ActiveSpace::n_active)
        .def("n_virtual", &ActiveSpace::n_virtual)
        .def("n_elec_active", &ActiveSpace::n_elec_active);

    py::class_<CASResult>(m, "CASResult")
        .def(py::init<>())
        .def_readwrite("e_casscf", &CASResult::e_casscf)
        .def_readwrite("e_nuclear", &CASResult::e_nuclear)
        .def_readwrite("n_iterations", &CASResult::n_iterations)
        .def_readwrite("converged", &CASResult::converged)
        .def_readwrite("C_mo", &CASResult::C_mo)
        .def_readwrite("orbital_energies", &CASResult::orbital_energies)
        .def_readwrite("ci_coeffs", &CASResult::ci_coeffs)
        .def_readwrite("determinants", &CASResult::determinants)
        .def_readwrite("n_determinants", &CASResult::n_determinants)
        .def_readwrite("active_space", &CASResult::active_space);

    py::class_<CASSCF>(m, "CASSCF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), py::arg("active_space"))
        .def("compute", &CASSCF::compute, py::arg("initial_guess"));

    py::class_<CASPT2Result>(m, "CASPT2Result")
        .def(py::init<>())
        .def_readwrite("e_casscf", &CASPT2Result::e_casscf)
        .def_readwrite("e_pt2", &CASPT2Result::e_pt2)
        .def_readwrite("e_total", &CASPT2Result::e_total)
        .def_readwrite("converged", &CASPT2Result::converged)
        .def_readwrite("status_message", &CASPT2Result::status_message);

    py::class_<CASPT2>(m, "CASPT2")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const CASResult&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), py::arg("casscf_result"))
        .def("compute", &CASPT2::compute);

    // ========================================================================
    // Gradient and Optimization
    // ========================================================================
    
    py::class_<GradientResult>(m, "GradientResult")
        .def(py::init<>())
        .def_readwrite("energy", &GradientResult::energy)
        .def_readwrite("gradient", &GradientResult::gradient)
        .def_readwrite("gradient_norm", &GradientResult::gradient_norm);

    py::class_<AnalyticalGradient>(m, "AnalyticalGradient")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"))
        .def("compute_rhf_gradient", &AnalyticalGradient::compute_rhf_gradient,
             py::arg("scf_result"));

    py::class_<OptConfig>(m, "OptConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &OptConfig::max_iterations)
        .def_readwrite("gradient_threshold", &OptConfig::gradient_threshold)
        .def_readwrite("energy_threshold", &OptConfig::energy_threshold)
        .def_readwrite("step_size", &OptConfig::step_size)
        .def_readwrite("print_level", &OptConfig::print_level);

    py::class_<OptResult>(m, "OptResult")
        .def(py::init<>())
        .def_readwrite("converged", &OptResult::converged)
        .def_readwrite("n_iterations", &OptResult::n_iterations)
        .def_readwrite("final_energy", &OptResult::final_energy)
        .def_readwrite("final_gradient_norm", &OptResult::final_gradient_norm)
        .def_readwrite("optimized_geometry", &OptResult::optimized_geometry);

    py::class_<GeometryOptimizer>(m, "GeometryOptimizer")
        .def(py::init<Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const OptConfig&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), 
             py::arg("config") = OptConfig())
        .def("optimize_rhf", &GeometryOptimizer::optimize_rhf);

    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    m.def("bohr_to_angstrom", [](double bohr) { return bohr * 0.529177210903; },
          "Convert Bohr to Angstrom");
    
    m.def("angstrom_to_bohr", [](double angstrom) { return angstrom / 0.529177210903; },
          "Convert Angstrom to Bohr");
    
    m.def("hartree_to_ev", [](double hartree) { return hartree * 27.211386245988; },
          "Convert Hartree to eV");
    
    m.def("hartree_to_kcal", [](double hartree) { return hartree * 627.5094740631; },
          "Convert Hartree to kcal/mol");
}
