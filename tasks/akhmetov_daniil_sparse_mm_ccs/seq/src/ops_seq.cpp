#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace akhmetov_daniil_sparse_mm_ccs {

bool SparseMatrixMultiplicationCCSSeq::ValidationImpl() {
  if (GetInput().size() != 2) {
    return false;
  }
  // a.cols == b.rows
  return GetInput()[0].cols == GetInput()[1].rows;
}

bool SparseMatrixMultiplicationCCSSeq::PreProcessingImpl() {
  res_matrix_ = SparseMatrixCCS();
  res_matrix_.rows = GetInput()[0].rows;
  res_matrix_.cols = GetInput()[1].cols;
  res_matrix_.col_ptr.assign(res_matrix_.cols + 1, 0);
  return true;
}

bool SparseMatrixMultiplicationCCSSeq::RunImpl() {
  const auto &a = GetInput()[0];
  const auto &b = GetInput()[1];
  std::vector<double> dense_col(a.rows, 0.0);

  for (int j = 0; j < b.cols; ++j) {
    std::fill(dense_col.begin(), dense_col.end(), 0.0);  // NOLINT

    for (int k_ptr = b.col_ptr[j]; k_ptr < b.col_ptr[j + 1]; ++k_ptr) {
      int k = b.row_indices[k_ptr];
      double val_b = b.values[k_ptr];

      for (int i_ptr = a.col_ptr[k]; i_ptr < a.col_ptr[k + 1]; ++i_ptr) {
        dense_col[a.row_indices[i_ptr]] += a.values[i_ptr] * val_b;
      }
    }

    for (int i = 0; i < a.rows; ++i) {
      if (std::abs(dense_col[i]) > 1e-15) {
        res_matrix_.values.push_back(dense_col[i]);
        res_matrix_.row_indices.push_back(i);
      }
    }
    res_matrix_.col_ptr[j + 1] = static_cast<int>(res_matrix_.values.size());
  }
  return true;
}

bool SparseMatrixMultiplicationCCSSeq::PostProcessingImpl() {
  GetOutput() = std::move(res_matrix_);
  return true;
}

}  // namespace akhmetov_daniil_sparse_mm_ccs
