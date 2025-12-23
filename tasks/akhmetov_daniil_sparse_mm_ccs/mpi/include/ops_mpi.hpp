#pragma once

#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akhmetov_daniil_sparse_mm_ccs {

class SparseMatrixMultiplicationCCSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SparseMatrixMultiplicationCCSMPI(const InType &in) {
    GetInput() = in;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastInputMatrices(int &rows_a, int &cols_a, int &cols_b, std::vector<int> &col_ptr_a,
                              std::vector<double> &values_a, std::vector<int> &rows_ind_a, std::vector<int> &col_ptr_b,
                              std::vector<double> &values_b, std::vector<int> &rows_ind_b);

  static void ComputeLocalProduct(int rank, int size, int rows_a, int cols_b, const std::vector<int> &col_ptr_a,
                                  const std::vector<double> &values_a, const std::vector<int> &rows_ind_a,
                                  const std::vector<int> &col_ptr_b, const std::vector<double> &values_b,
                                  const std::vector<int> &rows_ind_b, std::vector<double> &local_values,
                                  std::vector<int> &local_rows, std::vector<int> &local_col_ptr);

  void GatherResult(int rank, int size, int rows_a, int cols_b, const std::vector<double> &local_values,
                    const std::vector<int> &local_rows, const std::vector<int> &local_col_ptr);

  SparseMatrixCCS res_matrix_;
};

}  // namespace akhmetov_daniil_sparse_mm_ccs
