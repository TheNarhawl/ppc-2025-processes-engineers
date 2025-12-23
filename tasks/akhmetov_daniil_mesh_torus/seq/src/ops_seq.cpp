#include "akhmetov_daniil_mesh_torus/seq/include/ops_seq.hpp"

#include <vector>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akhmetov_daniil_mesh_torus {

MeshTorusSeq::MeshTorusSeq(const InType &in) {
  SetTypeOfTask(ppc::task::TypeOfTask::kSEQ);
  GetInput() = in;
}

bool MeshTorusSeq::ValidationImpl() {
  const auto &in = GetInput();
  return in.source >= 0 && in.dest >= 0;
}

bool MeshTorusSeq::PreProcessingImpl() {
  return true;
}

bool MeshTorusSeq::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  out.payload = in.payload;

  out.path.clear();
  out.path.push_back(in.source);
  if (in.source != in.dest) {
    out.path.push_back(in.dest);
  }

  return true;
}

bool MeshTorusSeq::PostProcessingImpl() {
  return true;
}

}  // namespace akhmetov_daniil_mesh_torus
