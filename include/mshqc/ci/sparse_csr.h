#ifndef MSHQC_CI_SPARSE_CSR_H
#define MSHQC_CI_SPARSE_CSR_H

#include <vector>
#include <cstdint>
#include <stdexcept>

namespace mshqc {
namespace ci {

class SparseCSR {
public:
    using Index = int32_t;
    using Scalar = double;

    void resize(Index n_rows, Index n_cols) {
        if (n_rows < 0 || n_cols < 0) throw std::invalid_argument("Negative dimension");
        n_rows_ = n_rows; n_cols_ = n_cols;
    }

    Index n_rows() const { return n_rows_; }
    Index n_cols() const { return n_cols_; }

    std::vector<Index>& row_ptr() { return row_ptr_; }
    std::vector<Index>& col_ind() { return col_ind_; }
    std::vector<Scalar>& values() { return values_; }
    const std::vector<Index>& row_ptr() const { return row_ptr_; }
    const std::vector<Index>& col_ind() const { return col_ind_; }
    const std::vector<Scalar>& values() const { return values_; }

    // Set CSR data by moving from provided vectors
    void set(Index n_rows, Index n_cols,
             std::vector<Index>&& rp,
             std::vector<Index>&& ci,
             std::vector<Scalar>&& vv) {
        resize(n_rows, n_cols);
        row_ptr_ = std::move(rp);
        col_ind_ = std::move(ci);
        values_  = std::move(vv);
        if (row_ptr_.size() != static_cast<size_t>(n_rows_) + 1)
            throw std::runtime_error("SparseCSR: row_ptr size mismatch");
        if (col_ind_.size() != values_.size())
            throw std::runtime_error("SparseCSR: col/val size mismatch");
    }

    // y = A x
    template <typename VecX, typename VecY>
    void matvec(const VecX& x, VecY& y) const {
        if (static_cast<Index>(x.size()) != n_cols_)
            throw std::runtime_error("SparseCSR::matvec: x size mismatch");
        y.assign(static_cast<size_t>(n_rows_), 0.0);
        for (Index r = 0; r < n_rows_; ++r) {
            const Index start = row_ptr_[static_cast<size_t>(r)];
            const Index end   = row_ptr_[static_cast<size_t>(r + 1)];
            double acc = 0.0;
            for (Index idx = start; idx < end; ++idx) {
                acc += values_[static_cast<size_t>(idx)] * x[static_cast<size_t>(col_ind_[static_cast<size_t>(idx)])];
            }
            y[static_cast<size_t>(r)] = acc;
        }
    }

private:
    Index n_rows_ = 0;
    Index n_cols_ = 0;
    std::vector<Index> row_ptr_;
    std::vector<Index> col_ind_;
    std::vector<Scalar> values_;
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_SPARSE_CSR_H
