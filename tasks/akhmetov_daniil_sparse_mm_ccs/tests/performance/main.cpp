#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

using akhmetov_daniil_sparse_mm_ccs::SparseMatrixCCS;
using akhmetov_daniil_sparse_mm_ccs::SparseMatrixMultiplicationCCSMPI;
using akhmetov_daniil_sparse_mm_ccs::SparseMatrixMultiplicationCCSSeq;
using InType = akhmetov_daniil_sparse_mm_ccs::InType;
using OutType = akhmetov_daniil_sparse_mm_ccs::OutType;

class SparseCCSPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kSize = 100000;

 protected:
  void SetUp() override {
    input_.clear();
    input_.push_back(MakeDiagonal(kSize, 1.0));
    input_.push_back(MakeDiagonal(kSize, 2.0));
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    EXPECT_EQ(out.rows, kSize);
    EXPECT_EQ(out.cols, kSize);
    EXPECT_EQ(out.col_ptr.size(), static_cast<size_t>(kSize + 1));
    EXPECT_EQ(out.values.size(), out.row_indices.size());
    return true;
  }

 private:
  InType input_;

  static SparseMatrixCCS MakeDiagonal(int n, double v) {
    SparseMatrixCCS m;
    m.rows = n;
    m.cols = n;
    m.col_ptr.resize(n + 1);
    for (int j = 0; j < n; ++j) {
      m.col_ptr[j] = static_cast<int>(m.values.size());
      m.values.push_back(v);
      m.row_indices.push_back(j);
    }
    m.col_ptr[n] = static_cast<int>(m.values.size());
    return m;
  }
};

using PerfParam = ppc::util::PerfTestParam<InType, OutType>;

static auto MakePerfParams() {
  return ppc::util::MakeAllPerfTasks<InType, SparseMatrixMultiplicationCCSSeq, SparseMatrixMultiplicationCCSMPI>("");
}

TEST_P(SparseCCSPerfTest, RunPerformance) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(CCSPerformance, SparseCCSPerfTest, ppc::util::TupleToGTestValues(MakePerfParams()),
                         SparseCCSPerfTest::CustomPerfTestName);
