#pragma once

#include <utility>
#include <vector>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"
#include "task/include/task.hpp"

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

  [[nodiscard]] static std::pair<int, int> ComputeGrid(int size);
  [[nodiscard]] static int RankFromCoords(int row, int col, int rows, int cols);
  [[nodiscard]] static std::pair<int, int> CoordsFromRank(int rank, int cols);
  [[nodiscard]] static std::vector<int> BuildPath(int rows, int cols, int source, int dest);

  void BroadcastSourceDest(int &source, int &dest);
  void BroadcastPayloadSize(int source, int &payload_size) const;
  [[nodiscard]] std::vector<int> PreparePayloadBuffer(int source, int payload_size) const;
  void ProcessPathCommunication(int source, int dest, const std::vector<int> &path, const std::vector<int> &payload_buf,
                                std::vector<int> &recv_payload) const;
  void SetOutput(int dest, const std::vector<int> &recv_payload, const std::vector<int> &path);

  InType local_in_{};
  OutType local_out_{};

  int world_rank_{0};
  int world_size_{0};

  int rows_{1};
  int cols_{1};
};

}  // namespace akhmetov_daniil_mesh_torus
