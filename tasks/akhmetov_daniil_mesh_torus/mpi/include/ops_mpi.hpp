#pragma once

#include <vector>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"

namespace akhmetov_daniil_mesh_torus {

class MeshTorusMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MeshTorusMpi(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] std::pair<int, int> ComputeGrid(int size) const;
  [[nodiscard]] int RankFromCoords(int row, int col, int rows, int cols) const;
  [[nodiscard]] std::pair<int, int> CoordsFromRank(int rank, int cols) const;
  [[nodiscard]] std::vector<int> BuildPath(int rows, int cols, int source, int dest) const;

  InType local_in_{};
  OutType local_out_{};

  int world_rank_{0};
  int world_size_{0};

  int rows_{1};
  int cols_{1};
};

}  // namespace akhmetov_daniil_mesh_torus
