#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include <algorithm>
#include <cmath>

namespace akhmetov_daniil_sparse_mm_ccs {

bool SparseMatrixMultiplicationCCSSeq::ValidationImpl() {
  if (GetInput().size() != 2) return false;
  // A.cols == B.rows
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
  const auto& A = GetInput()[0];
  const auto& B = GetInput()[1];
  
  std::vector<double> dense_col(A.rows, 0.0);

  for (int j = 0; j < B.cols; ++j) {
    std::fill(dense_col.begin(), dense_col.end(), 0.0);
    
    // Вычисляем столбец j матрицы C: C_j = A * B_j
    for (int k_ptr = B.col_ptr[j]; k_ptr < B.col_ptr[j + 1]; ++k_ptr) {
      int k = B.row_indices[k_ptr];
      double valB = B.values[k_ptr];

      for (int i_ptr = A.col_ptr[k]; i_ptr < A.col_ptr[k + 1]; ++i_ptr) {
        dense_col[A.row_indices[i_ptr]] += A.values[i_ptr] * valB;
      }
    }

    // Сохраняем ненулевые элементы в CCS
    for (int i = 0; i < A.rows; ++i) {
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