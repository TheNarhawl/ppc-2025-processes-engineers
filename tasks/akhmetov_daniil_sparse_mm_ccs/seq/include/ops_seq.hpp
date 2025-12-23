#pragma once

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akhmetov_daniil_sparse_mm_ccs {

class SparseMatrixMultiplicationCCSSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SparseMatrixMultiplicationCCSSeq(const InType &in) {
    GetInput() = in;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  SparseMatrixCCS res_matrix_;
};

}  // namespace akhmetov_daniil_sparse_mm_ccs
