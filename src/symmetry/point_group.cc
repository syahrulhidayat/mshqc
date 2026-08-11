// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

















#include "mshqc/symmetry/point_group.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/Eigenvalues>
#ifdef I
#undef I
#endif

namespace mshqc {

PointGroup::PointGroup(const Molecule& mol, double tolerance) : original_mol_(mol) {
    tolerance_ = tolerance;
    center_and_align();
}

void PointGroup::center_and_align() {

    int natoms = original_mol_.n_atoms();
    std::vector<Atom> temp_atoms;
    std::vector<double> masses(natoms);

    for(int i=0; i<natoms; ++i) {
        temp_atoms.push_back(original_mol_.atom(i));
        int Z = temp_atoms[i].atomic_number;
        masses[i] = (Z == 1) ? 1.0078 : (double)Z;
        if (Z == 8) masses[i] = 15.999;
    }

    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    double total_mass = 0.0;
    for(int i=0; i<natoms; ++i) {
        com += masses[i] * Eigen::Vector3d(temp_atoms[i].x, temp_atoms[i].y, temp_atoms[i].z);
        total_mass += masses[i];
    }
    com /= total_mass;

    for(int i=0; i<natoms; ++i) {
        temp_atoms[i].x -= com(0);
        temp_atoms[i].y -= com(1);
        temp_atoms[i].z -= com(2);
    }

    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for(int i=0; i<natoms; ++i) {
        double m = masses[i];
        double x = temp_atoms[i].x;
        double y = temp_atoms[i].y;
        double z = temp_atoms[i].z;
        inertia(0,0) += m * (y*y + z*z);
        inertia(1,1) += m * (x*x + z*z);
        inertia(2,2) += m * (x*x + y*y);
        inertia(0,1) -= m * x * y;
        inertia(0,2) -= m * x * z;
        inertia(1,2) -= m * y * z;
    }
    inertia(1,0) = inertia(0,1);
    inertia(2,0) = inertia(0,2);
    inertia(2,1) = inertia(1,2);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(inertia);
    Eigen::Matrix3d R = es.eigenvectors();
    if (R.determinant() < 0) R.col(2) *= -1;

    for(int i=0; i<natoms; ++i) {
        Eigen::Vector3d old_pos(temp_atoms[i].x, temp_atoms[i].y, temp_atoms[i].z);
        Eigen::Vector3d new_pos = R.transpose() * old_pos;
        if (std::abs(new_pos(0)) < tolerance_) new_pos(0) = 0.0;
        if (std::abs(new_pos(1)) < tolerance_) new_pos(1) = 0.0;
        if (std::abs(new_pos(2)) < tolerance_) new_pos(2) = 0.0;
        temp_atoms[i].x = new_pos(0);
        temp_atoms[i].y = new_pos(1);
        temp_atoms[i].z = new_pos(2);
    }

    Molecule new_mol;
    new_mol.set_charge(original_mol_.charge());
    new_mol.set_multiplicity(original_mol_.multiplicity());
    for(const auto& at : temp_atoms) new_mol.add_atom(at.atomic_number, at.x, at.y, at.z);
    aligned_mol_ = new_mol;
}

bool PointGroup::check_operation(const Eigen::Matrix3d& op_matrix) {
    int natoms = aligned_mol_.n_atoms();
    for (int i = 0; i < natoms; ++i) {
        Eigen::Vector3d pos_i(aligned_mol_.atom(i).x, aligned_mol_.atom(i).y, aligned_mol_.atom(i).z);
        int z_i = aligned_mol_.atom(i).atomic_number;
        Eigen::Vector3d pos_transformed = op_matrix * pos_i;
        bool match_found = false;
        for (int j = 0; j < natoms; ++j) {
            if (aligned_mol_.atom(j).atomic_number != z_i) continue;
            Eigen::Vector3d pos_j(aligned_mol_.atom(j).x, aligned_mol_.atom(j).y, aligned_mol_.atom(j).z);
            if ((pos_transformed - pos_j).norm() < tolerance_) {
                match_found = true; break;
            }
        }
        if (!match_found) return false;
    }
    return true;
}

bool PointGroup::has_inversion() { return check_operation(-1.0 * Eigen::Matrix3d::Identity()); }

bool PointGroup::has_c2(int axis) {
    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    if (axis == 0) { rot(1,1)=-1; rot(2,2)=-1; }
    if (axis == 1) { rot(0,0)=-1; rot(2,2)=-1; }
    if (axis == 2) { rot(0,0)=-1; rot(1,1)=-1; }
    return check_operation(rot);
}

bool PointGroup::has_sigma(int axis_normal) {
    Eigen::Matrix3d ref = Eigen::Matrix3d::Identity();
    ref(axis_normal, axis_normal) = -1;
    return check_operation(ref);
}

void add_d2h_ops(std::vector<SymmetryOperation>& ops) {
    Eigen::Matrix3d m;

    m = Eigen::Matrix3d::Identity(); ops.push_back({SymOpType::Identity, -1, m, "E"});

    m << -1,0,0, 0,-1,0, 0,0,1; ops.push_back({SymOpType::Rotation, 2, m, "C2(z)"});

    m << -1,0,0, 0,1,0, 0,0,-1; ops.push_back({SymOpType::Rotation, 2, m, "C2(y)"});

    m << 1,0,0, 0,-1,0, 0,0,-1; ops.push_back({SymOpType::Rotation, 2, m, "C2(x)"});

    m << -1,0,0, 0,-1,0, 0,0,-1; ops.push_back({SymOpType::Inversion, 1, m, "i"});

    m << 1,0,0, 0,1,0, 0,0,-1; ops.push_back({SymOpType::Reflection, 1, m, "s(xy)"});

    m << 1,0,0, 0,-1,0, 0,0,1; ops.push_back({SymOpType::Reflection, 1, m, "s(xz)"});

    m << -1,0,0, 0,1,0, 0,0,1; ops.push_back({SymOpType::Reflection, 1, m, "s(yz)"});
}

void PointGroup::detect() {
    operations_.clear();

    bool linear_z = true;
    for(int i=0; i<aligned_mol_.n_atoms(); ++i) {
        if (std::abs(aligned_mol_.atom(i).x) > tolerance_ ||
            std::abs(aligned_mol_.atom(i).y) > tolerance_) {
            linear_z = false; break;
        }
    }

    bool i_op = has_inversion();

    if (aligned_mol_.n_atoms() == 1) {
        symbol_ = "D2h";
        add_d2h_ops(operations_);
        return;
    }

    if (linear_z) {
        if (i_op) {
            symbol_ = "D2h";
            add_d2h_ops(operations_);
        } else {
            symbol_ = "C2v";

            operations_.push_back({SymOpType::Identity, -1, Eigen::Matrix3d::Identity(), "E"});
            Eigen::Matrix3d m;
            m << -1,0,0, 0,-1,0, 0,0,1; operations_.push_back({SymOpType::Rotation, 2, m, "C2(z)"});
            m << 1,0,0, 0,-1,0, 0,0,1; operations_.push_back({SymOpType::Reflection, 1, m, "s(xz)"});
            m << -1,0,0, 0,1,0, 0,0,1; operations_.push_back({SymOpType::Reflection, 1, m, "s(yz)"});
        }
        return;
    }

    bool c2x = has_c2(0); bool c2y = has_c2(1); bool c2z = has_c2(2);
    bool sig_yz = has_sigma(0); bool sig_xz = has_sigma(1); bool sig_xy = has_sigma(2);

    int n_c2 = (c2x?1:0) + (c2y?1:0) + (c2z?1:0);
    int n_sigma = (sig_yz?1:0) + (sig_xz?1:0) + (sig_xy?1:0);

    if (n_c2 == 3 && n_sigma == 3 && i_op) {
        symbol_ = "D2h";
        add_d2h_ops(operations_);
        return;
    }

    if (n_c2 == 1 && n_sigma == 2) {
        symbol_ = "C2v";
        operations_.push_back({SymOpType::Identity, -1, Eigen::Matrix3d::Identity(), "E"});

        if (c2z) {
            Eigen::Matrix3d m;
            m << -1,0,0, 0,-1,0, 0,0,1; operations_.push_back({SymOpType::Rotation, 2, m, "C2(z)"});

            if (sig_xz) { m << 1,0,0, 0,-1,0, 0,0,1; operations_.push_back({SymOpType::Reflection, 1, m, "s(xz)"}); }
            if (sig_yz) { m << -1,0,0, 0,1,0, 0,0,1; operations_.push_back({SymOpType::Reflection, 1, m, "s(yz)"}); }
        }
        else if (c2y) {
            Eigen::Matrix3d m;
            m << -1,0,0, 0,1,0, 0,0,-1; operations_.push_back({SymOpType::Rotation, 2, m, "C2(y)"});
            if (sig_xy) { m << 1,0,0, 0,1,0, 0,0,-1; operations_.push_back({SymOpType::Reflection, 1, m, "s(xy)"}); }
            if (sig_yz) { m << -1,0,0, 0,1,0, 0,0,1; operations_.push_back({SymOpType::Reflection, 1, m, "s(yz)"}); }
        }
        return;
    }

    symbol_ = "C1";
    operations_.push_back({SymOpType::Identity, -1, Eigen::Matrix3d::Identity(), "E"});
}

void PointGroup::find_abelian_subgroup() {}

}
