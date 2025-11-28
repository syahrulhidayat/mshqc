#ifndef MSHQC_CI_SPARSE_COO_H
#define MSHQC_CI_SPARSE_COO_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace mshqc {
namespace ci {

// Minimal COO sparse container for CI Hamiltonian construction.
// Day 1: Provide COO structure and finalize_to_csr; hooks for screening exist
// but actual screening policies will be integrated on Day 2.
class SparseCOO {
public:
    using Index = int32_t;
    using Scalar = double;

    SparseCOO(Index n_rows = 0, Index n_cols = 0)
        : n_rows_(n_rows), n_cols_(n_cols) {}

    void resize(Index n_rows, Index n_cols) {
        if (n_rows < 0 || n_cols < 0) throw std::invalid_argument("Negative dimension");
        n_rows_ = n_rows;
        n_cols_ = n_cols;
    }

    void reserve(std::size_t nnz) {
        rows_.reserve(nnz);
        cols_.reserve(nnz);
        vals_.reserve(nnz);
    }

    // Add a triplet (row, col, value) without screening
    void add(Index r, Index c, Scalar v) {
        bound_check_(r, c);
        rows_.push_back(r);
        cols_.push_back(c);
        vals_.push_back(v);
    }

    // Add a triplet only if predicate approves (screening hook)
    template <typename Pred>
    void add_if(Index r, Index c, Scalar v, const Pred& pred) {
        bound_check_(r, c);
        if (pred(r, c, v)) {
            rows_.push_back(r);
            cols_.push_back(c);
            vals_.push_back(v);
        }
    }

    std::size_t nnz() const { return vals_.size(); }

    void clear() {
        rows_.clear(); cols_.clear(); vals_.clear();
    }

    Index n_rows() const { return n_rows_; }
    Index n_cols() const { return n_cols_; }

    // Convert to CSR. If sum_duplicates=true, duplicates are summed.
    // Output arrays are overwritten.
    void finalize_to_csr(std::vector<Index>& row_ptr,
                         std::vector<Index>& col_ind,
                         std::vector<Scalar>& values,
                         bool sum_duplicates = true) const {
        const std::size_t nnz = vals_.size();
        if ((n_rows_ == 0 || n_cols_ == 0) && nnz > 0) {
            throw std::runtime_error("SparseCOO: non-empty with zero dimension");
        }

        // Build permutation to sort by (row, col)
        std::vector<std::size_t> perm(nnz);
        for (std::size_t i = 0; i < nnz; ++i) perm[i] = i;
        std::stable_sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
            if (rows_[a] != rows_[b]) return rows_[a] < rows_[b];
            return cols_[a] < cols_[b];
        });

        // Optionally combine duplicates while generating CSR
        row_ptr.assign(static_cast<std::size_t>(n_rows_) + 1, 0);
        col_ind.clear(); col_ind.reserve(nnz);
        values.clear();  values.reserve(nnz);

        Index current_row = 0;
        std::size_t i = 0;
        while (i < nnz) {
            // Advance row_ptr until we reach the row of perm[i]
            while (current_row < n_rows_ && (i == nnz || rows_[perm[i]] > current_row)) {
                row_ptr[static_cast<std::size_t>(current_row + 1)] = static_cast<Index>(col_ind.size());
                ++current_row;
            }
            if (i == nnz) break;

            const Index r = rows_[perm[i]];
            // Aggregate all entries in this row (and handle duplicates in that row)
            while (i < nnz && rows_[perm[i]] == r) {
                Index c = cols_[perm[i]];
                Scalar v = vals_[perm[i]];
                ++i;

                if (sum_duplicates) {
                    // Check if last entry has same column
                    if (!col_ind.empty() && row_ptr[r] < static_cast<Index>(col_ind.size()) && col_ind.back() == c) {
                        values.back() += v;
                    } else {
                        col_ind.push_back(c);
                        values.push_back(v);
                    }
                } else {
                    col_ind.push_back(c);
                    values.push_back(v);
                }

                // If next in same row has same column, the above logic will merge due to stable sort
                // Otherwise it will push a new (c, v)
            }

            row_ptr[static_cast<std::size_t>(r + 1)] = static_cast<Index>(col_ind.size());
            current_row = r + 1;
        }

        // Finish remaining empty rows, if any
        while (current_row < n_rows_) {
            row_ptr[static_cast<std::size_t>(current_row + 1)] = static_cast<Index>(col_ind.size());
            ++current_row;
        }
    }

private:
    void bound_check_(Index r, Index c) const {
        if (r < 0 || r >= n_rows_ || c < 0 || c >= n_cols_) {
            throw std::out_of_range("SparseCOO: index out of bounds");
        }
    }

    Index n_rows_ = 0;
    Index n_cols_ = 0;
    std::vector<Index> rows_;
    std::vector<Index> cols_;
    std::vector<Scalar> vals_;
};

} // namespace ci
} // namespace mshqc

#endif // MSHQC_CI_SPARSE_COO_H
