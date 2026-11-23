// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "mshqc/basis.h"
#include "mshqc/core/molecule.h"
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/utils/runtime_env.h"

namespace nb = nanobind;
using namespace mshqc;
using namespace mshqc::integrals;

namespace mshqc {
void bind_kernels_module(nanobind::module_& m);
} // namespace mshqc

NB_MODULE(_mshqc, m) {
    mshqc_auto_tune::initialize_hpc_environment();
    m.doc() = "MSHQC v2.0: Pure Thin-Core C++20 / Python Quantum Chemistry Engine";

    nb::class_<ERITensor>(m, "ERITensor")
        .def("size", &ERITensor::size)
        .def("dimension", &ERITensor::dimension)
        .def("shape", [](const ERITensor& t) {
            return std::make_tuple(t.dimension(0), t.dimension(1), t.dimension(2), t.dimension(3));
        })
        .def("__repr__", [](const ERITensor& t) {
            return "<mshqc.ERITensor shape=(" +
                   std::to_string(t.dimension(0)) + ", " +
                   std::to_string(t.dimension(1)) + ", " +
                   std::to_string(t.dimension(2)) + ", " +
                   std::to_string(t.dimension(3)) + ")>";
        });

    nb::class_<Atom>(m, "Atom")
        .def(nb::init<int, double, double, double>(),
             nb::arg("atomic_number"), nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def_rw("atomic_number", &Atom::atomic_number)
        .def_rw("x", &Atom::x)
        .def_rw("y", &Atom::y)
        .def_rw("z", &Atom::z)
        .def("position", &Atom::position);

    nb::class_<Molecule>(m, "Molecule")
        .def(nb::init<int, int>(), nb::arg("charge") = 0, nb::arg("multiplicity") = 1)
        .def("add_atom", [](Molecule& mol, int Z, double x, double y, double z, const std::string& units) {
            double s = mshqc_auto_tune::unit_scale(units);
            mol.add_atom(Z, x * s, y * s, z * s);
        }, nb::arg("Z"), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("units") = "bohr")
        .def("add_atom", [](Molecule& mol, const std::string& symbol, double x, double y, double z, const std::string& units) {
            double s = mshqc_auto_tune::unit_scale(units);
            int Z = mshqc_auto_tune::symbol_to_z(symbol);
            mol.add_atom(Z, x * s, y * s, z * s);
        }, nb::arg("symbol"), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("units") = "bohr")
        .def("add_atom", nb::overload_cast<const Atom&>(&Molecule::add_atom), nb::arg("atom"))
        .def("n_atoms", &Molecule::n_atoms)
        .def("atom", &Molecule::atom, nb::arg("i"))
        .def("atoms", &Molecule::atoms)
        .def("total_nuclear_charge", &Molecule::total_nuclear_charge)
        .def("n_electrons", &Molecule::n_electrons)
        .def("charge", &Molecule::charge)
        .def("set_charge", &Molecule::set_charge, nb::arg("q"))
        .def("multiplicity", &Molecule::multiplicity)
        .def("set_multiplicity", &Molecule::set_multiplicity, nb::arg("m"))
        .def("nuclear_repulsion_energy", &Molecule::nuclear_repulsion_energy)
        .def("set_atom_position", &Molecule::set_atom_position,
             nb::arg("i"), nb::arg("x"), nb::arg("y"), nb::arg("z"));

    nb::enum_<AngularMomentum>(m, "AngularMomentum")
        .value("S", AngularMomentum::S)
        .value("P", AngularMomentum::P)
        .value("D", AngularMomentum::D)
        .value("F", AngularMomentum::F)
        .value("G", AngularMomentum::G)
        .value("H", AngularMomentum::H)
        .value("I", AngularMomentum::I)
        .value("K", AngularMomentum::K);

    nb::class_<GaussianPrimitive>(m, "GaussianPrimitive")
        .def(nb::init<double, double>(), nb::arg("exponent"), nb::arg("coefficient"))
        .def_rw("exponent", &GaussianPrimitive::exponent)
        .def_rw("coefficient", &GaussianPrimitive::coefficient);

    nb::class_<Shell>(m, "Shell")
        .def(nb::init<AngularMomentum, int, const std::array<double, 3>&>(),
             nb::arg("am"), nb::arg("center"), nb::arg("center_pos"))
        .def_prop_ro("angular_momentum", &Shell::angular_momentum)
        .def_prop_ro("center", &Shell::center)
        .def_prop_ro("primitives", &Shell::primitives)
        .def("l", &Shell::l)
        .def("n_functions", &Shell::n_functions);

    nb::class_<BasisSet>(m, "BasisSet")
        .def(nb::init<>())
        .def(nb::init<const std::string&, const Molecule&, const std::string&>(),
             nb::arg("basis_name"), nb::arg("mol"), nb::arg("basis_dir") = "data/basis")
        .def("read_gbs", &BasisSet::read_gbs, nb::arg("basis_file"), nb::arg("mol"))
        .def("add_shell", &BasisSet::add_shell, nb::arg("shell"))
        .def("n_shells", &BasisSet::n_shells)
        .def("n_basis_functions", &BasisSet::n_basis_functions)
        .def("n_basis", &BasisSet::n_basis_functions)
        .def("shell", &BasisSet::shell, nb::arg("i"))
        .def("name", &BasisSet::name);

    nb::class_<IntegralEngine>(m, "IntegralEngine")
        .def(nb::init<const Molecule&, const BasisSet&>(), nb::arg("mol"), nb::arg("basis"))
        .def("compute_overlap", &IntegralEngine::compute_overlap, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_kinetic", &IntegralEngine::compute_kinetic, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_nuclear", &IntegralEngine::compute_nuclear, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_eri", &IntegralEngine::compute_eri, nb::rv_policy::reference_internal, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_eri_diagonal", &IntegralEngine::compute_eri_diagonal)
        .def("compute_eri_column", &IntegralEngine::compute_eri_column, nb::arg("pivot_index"));

    nb::class_<CholeskyDecompositionResult>(m, "CholeskyDecompositionResult")
        .def_ro("n_vectors", &CholeskyDecompositionResult::n_vectors)
        .def_ro("n_basis", &CholeskyDecompositionResult::n_basis)
        .def_ro("threshold", &CholeskyDecompositionResult::threshold)
        .def_ro("compression_ratio", &CholeskyDecompositionResult::compression_ratio)
        .def_ro("converged", &CholeskyDecompositionResult::converged);

    nb::class_<CholeskyERI>(m, "CholeskyERI")
        .def(nb::init<double>(), nb::arg("threshold") = 1e-9)
        .def(nb::init<const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("basis"), nb::arg("integrals"))
        .def("__init__", [](CholeskyERI* chol, const Molecule& mol, const BasisSet& basis, double threshold) {
            auto ints = std::make_shared<IntegralEngine>(mol, basis);
            new (chol) CholeskyERI(basis, ints);
            chol->set_threshold(threshold);
            chol->compute();
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("threshold") = 1e-9)
        .def("set_threshold", &CholeskyERI::set_threshold)
        .def("set_print_level", &CholeskyERI::set_print_level)
        .def("n_vectors", &CholeskyERI::n_vectors)
        .def("n_cholesky_vectors", &CholeskyERI::n_vectors)
        .def("threshold", &CholeskyERI::threshold)
        .def("compute", &CholeskyERI::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("decompose", &CholeskyERI::decompose, nb::call_guard<nb::gil_scoped_release>())
        .def("get_L_mat", &CholeskyERI::get_L_mat, nb::rv_policy::reference_internal)
        .def("reconstruct", &CholeskyERI::reconstruct)
        .def("reconstruct_full", &CholeskyERI::reconstruct_full)
        .def("is_decomposed", &CholeskyERI::is_decomposed)
        .def("decomposed", &CholeskyERI::decomposed);

    m.def("bohr_to_angstrom", [](double bohr) { return bohr * 0.529177210903; }, "Convert Bohr to Angstrom");
    m.def("angstrom_to_bohr", [](double angstrom) { return angstrom / 0.529177210903; }, "Convert Angstrom to Bohr");
    m.def("hartree_to_ev", [](double hartree) { return hartree * 27.211386245988; }, "Convert Hartree to eV");
    m.def("hartree_to_kcal", [](double hartree) { return hartree * 627.5094740631; }, "Convert Hartree to kcal/mol");
    m.def("standard_frozen_core", &mshqc_auto_tune::standard_frozen_core_count, nb::arg("mol"),
          "Compute standard noble-gas frozen core orbital count (Psi4 convention)");

    mshqc::bind_kernels_module(m);
}
