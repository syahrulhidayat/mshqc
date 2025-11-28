#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <mshqc/molecule.h>
#include <mshqc/basis.h>
#include <mshqc/scf.h>
#include <mshqc/mp2.h>
#include <mshqc/ci.h>
#include <mshqc/mcscf.h>
#include <mshqc/wavefunction.h>

namespace py = pybind11;
using namespace mshqc;

// Helper function to create numpy array from Eigen::MatrixXd
py::array_t<double> eigen_to_numpy(const Eigen::MatrixXd& matrix) {
    return py::array_t<double>(
        py::buffer_info(
            matrix.data(),                                 /* Pointer to data */
            sizeof(double),                                /* Size of one item */
            py::format_descriptor<double>::format(),        /* Format */
            2,                                              /* Number of dimensions */
            { matrix.rows(), matrix.cols() },               /* Shape of array */
            { sizeof(double) * matrix.cols(), sizeof(double) }  /* Strides */
        )
    );
}

// Helper function to create numpy array from Eigen::VectorXd
py::array_t<double> eigen_to_numpy(const Eigen::VectorXd& vector) {
    return py::array_t<double>(
        py::buffer_info(
            vector.data(),                                 /* Pointer to data */
            sizeof(double),                                 /* Size of one item */
            py::format_descriptor<double>::format(),         /* Format */
            1,                                               /* Number of dimensions */
            { vector.rows() },                              /* Shape of array */
            { sizeof(double) }                              /* Strides */
        )
    );
}

// Helper function to get MO coefficients from SCFResult
py::array_t<double> get_mo_coeffs(const SCFResult& result) {
    return eigen_to_numpy(result.C);
}

// Helper function to get orbital energies from SCFResult
py::array_t<double> get_orbital_energies(const SCFResult& result) {
    return eigen_to_numpy(result.e);
}

// Helper function to get density matrix from SCFResult
py::array_t<double> get_density_matrix(const SCFResult& result) {
    return eigen_to_numpy(result.D);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "MSH-QC: Quantum Mechanics Library for Computational Chemistry";

    py::class_<Molecule>(m, "Molecule")
        .def(py::init<>())
        .def("add_atom", &Molecule::add_atom,
             py::arg("atomic_number"), py::arg("x"), py::arg("y"), py::arg("z"),
             "Add an atom to the molecule")
        .def("natoms", &Molecule::natoms, "Number of atoms")
        .def("nelectrons", &Molecule::nelectrons, "Number of electrons")
        .def("charge", &Molecule::charge, "Total charge")
        .def("multiplicity", &Molecule::multiplicity, "Spin multiplicity")
        .def("get_atomic_numbers", [](const Molecule& mol) {
            return mol.get_atomic_numbers();
        }, "Get atomic numbers")
        .def("get_coordinates", [](const Molecule& mol) {
            return eigen_to_numpy(mol.get_coordinates());
        }, "Get atomic coordinates");

    py::class_<BasisSet>(m, "BasisSet")
        .def(py::init<>())
        .def("load_minimal", &BasisSet::load_minimal,
             py::arg("basis_name"),
             "Load a minimal basis set")
        .def("load", &BasisSet::load,
             py::arg("basis_name"), py::arg("data_path"),
             "Load a basis set from file")
        .def("nbasis", &BasisSet::nbasis, "Number of basis functions")
        .def("nbf", &BasisSet::nbf, "Number of basis functions (alias)")
        .def("max_L", &BasisSet::max_L, "Maximum angular momentum");

    py::class_<SCFResult>(m, "SCFResult")
        .def(py::init<>())
        .def_readonly("energy", &SCFResult::energy, "Total energy")
        .def_property("C", &get_mo_coeffs, "MO coefficients")
        .def_property("e", &get_orbital_energies, "Orbital energies")
        .def_property("D", &get_density_matrix, "Density matrix")
        .def_readonly("n_occ", &SCFResult::n_occ, "Number of occupied orbitals")
        .def_readonly("n_vir", &SCFResult::n_vir, "Number of virtual orbitals")
        .def_readonly("converged", &SCFResult::converged, "Convergence status");

    py::class_<RHF>(m, "RHF")
        .def(py::init<const Molecule&, const BasisSet&>(),
             py::arg("molecule"), py::arg("basis"))
        .def("solve", &RHF::solve, 
             "Solve RHF equations")
        .def("get_energy", &RHF::get_energy,
             "Get current energy")
        .def("get_density_matrix", [](const RHF& rhf) {
            return eigen_to_numpy(rhf.get_density_matrix());
        }, "Get density matrix");

    py::class_<UHF>(m, "UHF")
        .def(py::init<const Molecule&, const BasisSet&>(),
             py::arg("molecule"), py::arg("basis"))
        .def("solve", &UHF::solve,
             "Solve UHF equations")
        .def("get_energy", &UHF::get_energy,
             "Get current energy")
        .def("get_alpha_density", [](const UHF& uhf) {
            return eigen_to_numpy(uhf.get_alpha_density());
        }, "Get alpha density matrix")
        .def("get_beta_density", [](const UHF& uhf) {
            return eigen_to_numpy(uhf.get_beta_density());
        }, "Get beta density matrix");

    py::class_<MP2>(m, "MP2")
        .def(py::init<const SCFResult&>(),
             py::arg("ref_result"))
        .def("solve", &MP2::solve,
             "Solve MP2 equations")
        .def("get_energy", &MP2::get_energy,
             "Get MP2 energy")
        .def("get_correlation_energy", &MP2::get_correlation_energy,
             "Get MP2 correlation energy");

    py::class_<CISD>(m, "CISD")
        .def(py::init<const SCFResult&>(),
             py::arg("ref_result"))
        .def("solve", &CISD::solve,
             "Solve CISD equations")
        .def("get_energy", &CISD::get_energy,
             "Get CISD energy")
        .def("get_correlation_energy", &CISD::get_correlation_energy,
             "Get CISD correlation energy");

    py::class_<CASSCF>(m, "CASSCF")
        .def(py::init<const Molecule&, const BasisSet&>(),
             py::arg("molecule"), py::arg("basis"))
        .def("set_active_space", &CASSCF::set_active_space,
             py::arg("n_electrons"), py::arg("n_orbitals"),
             "Set active space")
        .def("solve", &CASSCF::solve,
             "Solve CASSCF equations")
        .def("get_energy", &CASSCF::get_energy,
             "Get CASSCF energy");

    // Utility functions
    m.def("create_water_molecule", []() {
        Molecule mol;
        mol.add_atom(8, 0.0, 0.0, 0.119262);    // O
        mol.add_atom(1, 0.0, 0.757, -0.477023);  // H
        mol.add_atom(1, 0.0, -0.757, -0.477023); // H
        return mol;
    }, "Create a water molecule");

    m.def("create_h2_molecule", [](double distance = 0.74) {
        Molecule mol;
        mol.add_atom(1, 0.0, 0.0, -distance/2.0);  // H
        mol.add_atom(1, 0.0, 0.0, distance/2.0);   // H
        return mol;
    }, "Create a hydrogen molecule", py::arg("distance") = 0.74);

    m.def("create_li_atom", []() {
        Molecule mol;
        mol.add_atom(3, 0.0, 0.0, 0.0);  // Li
        return mol;
    }, "Create a lithium atom");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}