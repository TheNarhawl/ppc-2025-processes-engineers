#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace akhmetov_daniil_sparse_mm_ccs {

struct SparseMatrixCCS {
  int rows = 0;
  int cols = 0;
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_ptr;

  bool operator==(const SparseMatrixCCS &other) const {
    return rows == other.rows && cols == other.cols && values == other.values && row_indices == other.row_indices &&
           col_ptr == other.col_ptr;
  }
};

using InType = std::vector<SparseMatrixCCS>;
using OutType = SparseMatrixCCS;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace akhmetov_daniil_sparse_mm_ccs
