#ifndef MSHQC_SYMMETRY_PETITE_LIST_H
#define MSHQC_SYMMETRY_PETITE_LIST_H

#include "mshqc/basis.h"
#include "mshqc/symmetry/point_group.h"
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

// --- STRUKTUR 2D (Untuk UHF, ROHF, MP3 In-Core) ---
struct UniqueShellPair {
    int p; 
    int q; 
    double weight; 
};

// --- STRUKTUR 4D (Untuk Direct SCF RHF) ---
struct UniqueShellQuartet {
    int M; 
    int N; 
    int P; 
    int Q; 
    double weight; 
};

class PetiteList {
public:
    PetiteList(const BasisSet& basis, const PointGroup& pg);
    
    void build();
    
    // Getter untuk 2D dan 4D
    const std::vector<UniqueShellPair>& get_unique_pairs() const { return unique_pairs_; }
    const std::vector<UniqueShellQuartet>& get_unique_quartets() const { return unique_quartets_; }

private:
    const BasisSet& basis_;
    const PointGroup& pg_;
    
    std::vector<UniqueShellPair> unique_pairs_;
    std::vector<UniqueShellQuartet> unique_quartets_;
    
    int find_shell_at(const Eigen::Vector3d& pos, int original_shell_idx) const;
};

} // namespace mshqc
#endif