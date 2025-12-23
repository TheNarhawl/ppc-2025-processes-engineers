#pragma once

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akhmetov_daniil_mesh_torus {

class MeshTorusSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit MeshTorusSeq(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType local_in_{};
  OutType local_out_{};
};

}  // namespace akhmetov_daniil_mesh_torus
