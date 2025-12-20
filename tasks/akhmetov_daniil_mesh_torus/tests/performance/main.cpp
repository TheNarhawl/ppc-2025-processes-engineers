#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"
#include "akhmetov_daniil_mesh_torus/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_mesh_torus/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akhmetov_daniil_mesh_torus {

using ppc::util::PerfTestParam;

class MeshTorusPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType test_input_data_;
  bool data_prepared_ = false;
  int world_size_ = 1;
  int rank_ = 0;
  bool is_seq_test_ = false;

  void SetUp() override {
    std::string task_name = std::get<1>(GetParam());
    is_seq_test_ = (task_name.find("seq") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    }

    PrepareTestData();
  }

  void PrepareTestData() {
    if (data_prepared_) {
      return;
    }

    const int data_size = 10000000;

    test_input_data_.source = 0;
    test_input_data_.dest = is_seq_test_ ? 0 : (world_size_ > 1 ? (world_size_ - 1) : 0);

    test_input_data_.payload.resize(data_size);
    for (int i = 0; i < data_size; ++i) {
      test_input_data_.payload[i] = i + 1;
    }

    data_prepared_ = true;
  }

  InType GetTestInputData() override {
    return test_input_data_;
  }

  bool CheckTestOutputData(OutType &out) override {
    std::string task_name = std::get<1>(GetParam());
    bool is_mpi = (task_name.find("mpi") != std::string::npos);

    if (is_mpi) {
      if (rank_ != test_input_data_.dest) {
        return out.payload.empty();
      }
      if (out.payload.size() != test_input_data_.payload.size()) {
        return false;
      }
      if (out.payload.empty()) {
        return true;
      }
      return out.payload.front() == test_input_data_.payload.front() &&
             out.payload.back() == test_input_data_.payload.back();
    } else {
      if (rank_ == 0) {
        if (out.payload.size() != test_input_data_.payload.size()) {
          return false;
        }
        if (out.payload.empty()) {
          return true;
        }
        return out.payload.front() == test_input_data_.payload.front() &&
               out.payload.back() == test_input_data_.payload.back();
      }
      return true;
    }
  }
};

namespace {
const auto kPerfTasksTuples =
    ppc::util::MakeAllPerfTasks<InType, MeshTorusMpi, MeshTorusSeq>(PPC_SETTINGS_akhmetov_daniil_mesh_torus);

const auto kPerfValues = ppc::util::TupleToGTestValues(kPerfTasksTuples);
const auto kPerfNamePrinter = MeshTorusPerfTest::CustomPerfTestName;

TEST_P(MeshTorusPerfTest, MeshTorusPerformance) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(MeshTorusPerf, MeshTorusPerfTest, kPerfValues, kPerfNamePrinter);

}  // namespace
}  // namespace akhmetov_daniil_mesh_torus
