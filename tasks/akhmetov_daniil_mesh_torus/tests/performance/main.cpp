#include <gtest/gtest.h>
#include <mpi.h>

#include <string>

#include "akhmetov_daniil_mesh_torus/common/include/common.hpp"
#include "akhmetov_daniil_mesh_torus/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_mesh_torus/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akhmetov_daniil_mesh_torus {

using ppc::util::PerfTestParam;

class MeshTorusPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType test_input_data;
  bool data_prepared = false;
  int world_size = 1;
  int rank = 0;
  bool is_seq_test = false;

  void SetUp() override {
    std::string task_name = std::get<1>(GetParam());
    is_seq_test = (task_name.find("seq") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    PrepareTestData();
  }

  void PrepareTestData() {
    if (data_prepared) {
      return;
    }

    const int data_size = 10000000;

    test_input_data.source = 0;
    if (is_seq_test) {
      test_input_data.dest = 0;
    } else {
      if (world_size > 1) {
        test_input_data.dest = world_size - 1;
      } else {
        test_input_data.dest = 0;
      }
    }

    test_input_data.payload.resize(data_size);
    for (int i = 0; i < data_size; ++i) {
      test_input_data.payload[i] = i + 1;
    }

    data_prepared = true;
  }

  InType GetTestInputData() override {
    return test_input_data;
  }

  bool CheckTestOutputData(OutType &out) override {
    std::string task_name = std::get<1>(GetParam());
    bool is_mpi = (task_name.find("mpi") != std::string::npos);

    if (is_mpi) {
      if (rank != test_input_data.dest) {
        return out.payload.empty();
      }
      if (out.payload.size() != test_input_data.payload.size()) {
        return false;
      }
      if (out.payload.empty()) {
        return true;
      }
      return out.payload.front() == test_input_data.payload.front() &&
             out.payload.back() == test_input_data.payload.back();
    }

    if (rank == 0) {
      if (out.payload.size() != test_input_data.payload.size()) {
        return false;
      }
      if (out.payload.empty()) {
        return true;
      }
      return out.payload.front() == test_input_data.payload.front() &&
             out.payload.back() == test_input_data.payload.back();
    }
    return true;
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
