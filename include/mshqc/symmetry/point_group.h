#ifndef MSHQC_SYMMETRY_POINT_GROUP_H
#define MSHQC_SYMMETRY_POINT_GROUP_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "mshqc/core/molecule.h"
#ifdef I
#undef I
#endif

namespace mshqc {

struct Irrep {
    int id;               

    std::string name;     

};

class CharacterTable {
public:
    std::string group_name;
    std::vector<Irrep> irreps;
    Eigen::MatrixXd characters; 


    

    int direct_product(int irrep1, int irrep2) const {
        return irrep1 ^ irrep2; 

    }
};



enum class SymOpType { 
    Identity, Rotation, Reflection, Inversion, ImproperRotation 
};
struct SymmetryOperation {
    SymOpType type;
    int order;              
    Eigen::Matrix3d matrix; 
    std::string name;       
};

class PointGroup {
public:
    PointGroup(const Molecule& mol);

    void detect();
    CharacterTable get_character_table() const;
    

    const std::vector<SymmetryOperation>& get_operations() const { return operations_; }
    
    

    std::string symbol() const { return symbol_; }
    std::string get_symbol() const { return symbol_; }
    
    int get_order() const { return operations_.size(); }
    const Molecule& get_aligned_molecule() const { return aligned_mol_; }


private:
    Molecule original_mol_;
    Molecule aligned_mol_;
    std::string symbol_;
    std::vector<SymmetryOperation> operations_;
    double tolerance_;

    void center_and_align();
    bool check_operation(const Eigen::Matrix3d& op_matrix);
    bool has_inversion();
    bool has_c2(int axis);
    bool has_sigma(int axis_normal);
    void find_abelian_subgroup();
};

} 


#endif 
