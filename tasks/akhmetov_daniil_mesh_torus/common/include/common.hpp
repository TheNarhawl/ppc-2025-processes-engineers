#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace akhmetov_daniil_mesh_torus {

struct InputData {
  int source{};
  int dest{};
  std::vector<int> payload;
};

struct OutputData {
  std::vector<int> payload;
  std::vector<int> path;
};

using InType = InputData;
using OutType = OutputData;
using TestType = std::tuple<int>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace akhmetov_daniil_mesh_torus
