#ifndef MSHQC_INTEGRALS_CHOLESKY_ERI_H
#define MSHQC_INTEGRALS_CHOLESKY_ERI_H

#include "mshqc/basis.h"
#include "mshqc/integrals.h" 
#include <vector>
#include <utility>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace integrals {

// Hasil dekomposisi untuk report
struct CholeskyDecompositionResult {
    int n_vectors = 0;
    int n_basis = 0;
    double threshold = 0.0;
    double max_error = 0.0;
    double rms_error = 0.0;
    double compression_ratio = 0.0;
    bool converged = false;
};

// Konfigurasi
struct CholeskyERIConfig {
    double threshold = 1e-6;
    int print_level = 1;
};

class CholeskyERI {
public:
    // Constructor
    explicit CholeskyERI(double threshold = 1e-6);
    CholeskyERI(const BasisSet& basis, std::shared_ptr<IntegralEngine> integrals);

    ~CholeskyERI() = default;
    
    // --- MAIN METHODS ---
    void compute(); 
    void decompose_direct();
    CholeskyDecompositionResult decompose(const Eigen::Tensor<double, 4>& eri_full);

    // --- RECONSTRUCTION ---
    double reconstruct(int i, int j, int k, int l) const;
    Eigen::Tensor<double, 4> reconstruct_full() const;
    
    // --- SETTERS & GETTERS ---
    void set_threshold(double t) { config_.threshold = t; threshold_ = t; }
    void set_print_level(int p) { config_.print_level = p; }
    
    double threshold() const { return threshold_; }
    int n_basis() const { return n_basis_; }
    int n_vectors() const { return n_vectors_; }
    
    //const std::vector<Eigen::VectorXd>& get_L_vectors() const { return L_vectors_; }
    
    // Alias untuk kompatibilitas
    int get_n_vectors() const { return n_vectors_; }
    //          const std::vector<Eigen::VectorXd>& get_vectors() const { return L_vectors_; }

    // [FIX] Menambahkan decomposed() agar kompatibel dengan UMP3
    bool decomposed() const { return decomposed_; }
    bool is_decomposed() const { return decomposed_; }
    
    double compression_ratio() const;
    size_t storage_bytes() const;
    
    void reset();
    const Eigen::MatrixXd& get_L_mat() const { return L_mat_; }
    const Eigen::MatrixXd& get_L_matrix() const { return L_mat_; }
    const std::vector<Eigen::MatrixXd>& get_L_vectors() const {
        return L_vectors_; // Pastikan nama variabel internalnya benar
    }

    // --- LEGACY STUBS (Agar bindings.cc tidak error) ---
    std::pair<double, double> validate_reconstruction(const Eigen::Tensor<double, 4>& eri_exact);
    void print_statistics(bool verbose = false) const;
    int find_pivot(const Eigen::VectorXd& D) const;
    Eigen::VectorXd compute_new_vector(const Eigen::Tensor<double, 4>& eri, const Eigen::VectorXd& D, int pivot) const;
    /**
     * @brief Menyimpan vektor Cholesky ke SSD menggunakan HDF5.
     */
    void save_to_hdf5(const std::string& filename, const std::string& dataset_name = "cholesky_vectors") const;

    /**
     * @brief Memuat vektor Cholesky dari file HDF5 di SSD ke RAM.
     */
    void load_from_hdf5(const std::string& filename, const std::string& dataset_name = "cholesky_vectors");
    

private:
    // Pointers
    const BasisSet* basis_ptr_ = nullptr;
    std::shared_ptr<IntegralEngine> integrals_ptr_;

    // Config & Storage
    CholeskyERIConfig config_; 
    Eigen::MatrixXd L_mat_;
    std::vector<int> bf2shell_;
    std::vector<Eigen::MatrixXd> L_vectors_; 
    
    
    int n_basis_ = 0;
    int n_vectors_ = 0;
    double threshold_ = 1e-6;
    bool decomposed_ = false;
    double max_error_ = 0.0;
    double rms_error_ = 0.0;
};

} // namespace integrals
} // namespace mshqc

#endif // MSHQC_INTEGRALS_CHOLESKY_ERI_H