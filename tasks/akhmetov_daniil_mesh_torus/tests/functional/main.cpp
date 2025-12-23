#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"
#include "akhmetov_daniil_mesh_torus/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_mesh_torus/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace akhmetov_daniil_mesh_torus {

using ppc::util::FuncTestParam;

class MeshTorusFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int n = std::get<0>(test_param);
    return "n" + std::to_string(n);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<2>(GetParam());
    int data_size = std::get<0>(params);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    } else {
      world_size_ = 1;
      rank_ = 0;
    }

    source_ = 0;
    dest_ = (world_size_ > 1) ? (world_size_ - 1) : 0;

    input_.source = source_;
    input_.dest = dest_;
    input_.payload.clear();
    for (int i = 0; i < data_size; ++i) {
      input_.payload.push_back(i + 1);
    }
    expected_payload_ = input_.payload;
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    const std::string &task_name = std::get<1>(GetParam());
    const bool is_seq = (task_name.find("seq") != std::string::npos);

    return is_seq ? CheckSeq(out) : CheckMpi(out);
  }

 private:
  [[nodiscard]] bool CheckSeq(const OutType &out) const {
    if (out.payload != expected_payload_) {
      return false;
    }
    if (out.path.empty()) {
      return false;
    }
    if (out.path.front() != source_) {
      return false;
    }
    if (out.path.back() != dest_) {
      return false;
    }
    return true;
  }

  [[nodiscard]] bool CheckMpi(const OutType &out) const {
    if (rank_ != dest_) {
      return out.payload.empty() && out.path.empty();
    }
    if (out.payload != expected_payload_) {
      return false;
    }
    if (out.path.empty()) {
      return false;
    }
    if (out.path.front() != source_) {
      return false;
    }
    if (out.path.back() != dest_) {
      return false;
    }
    return true;
  }

  InType input_{};
  std::vector<int> expected_payload_;
  int world_size_{1};
  int rank_{0};
  int source_{0};
  int dest_{0};
};

namespace {

const std::array<TestType, 5> kTestParams = {
    std::make_tuple(1), std::make_tuple(4), std::make_tuple(8), std::make_tuple(16), std::make_tuple(32),
};

const auto kTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<MeshTorusMpi, InType>(kTestParams, PPC_SETTINGS_akhmetov_daniil_mesh_torus),
                   ppc::util::AddFuncTask<MeshTorusSeq, InType>(kTestParams, PPC_SETTINGS_akhmetov_daniil_mesh_torus));

const auto kValues = ppc::util::ExpandToValues(kTasksList);
const auto kNamePrinter = MeshTorusFuncTest::PrintFuncTestName<MeshTorusFuncTest>;

TEST_P(MeshTorusFuncTest, MeshTorusDataTransfer) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(MeshTorusTests, MeshTorusFuncTest, kValues, kNamePrinter);

}  // namespace
}  // namespace akhmetov_daniil_mesh_torus
