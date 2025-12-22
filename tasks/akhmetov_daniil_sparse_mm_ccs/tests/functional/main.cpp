// tests/functional/main.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

using akhmetov_daniil_sparse_mm_ccs::SparseMatrixCCS;
using akhmetov_daniil_sparse_mm_ccs::SparseMatrixMultiplicationCCSMPI;
using akhmetov_daniil_sparse_mm_ccs::SparseMatrixMultiplicationCCSSeq;
using InType = akhmetov_daniil_sparse_mm_ccs::InType;
using OutType = akhmetov_daniil_sparse_mm_ccs::OutType;

static SparseMatrixCCS FromDense(const std::vector<std::vector<double>> &dense) {
  SparseMatrixCCS m;
  m.rows = static_cast<int>(dense.size());
  m.cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  m.col_ptr.assign(m.cols + 1, 0);

  for (int j = 0; j < m.cols; ++j) {
    for (int i = 0; i < m.rows; ++i) {
      double v = dense[i][j];
      if (std::abs(v) > 1e-12) {
        m.values.push_back(v);
        m.row_indices.push_back(i);
      }
    }
    m.col_ptr[j + 1] = static_cast<int>(m.values.size());
  }
  return m;
}

struct TestCase {
  std::vector<std::vector<double>> A;
  std::vector<std::vector<double>> B;
  std::string name;
};

namespace {
const TestCase kSmallExact = {{{1.0, 0.0, 2.0}, {0.0, 3.0, 0.0}}, {{0.0, 4.0}, {5.0, 0.0}, {0.0, 6.0}}, "SmallExact"};

TestCase GenerateRandomMedium() {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> val(-10.0, 10.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  auto gen_dense_matrix = [&](int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if (prob(gen) < 0.05) {
          matrix[i][j] = val(gen);
        }
      }
    }
    return matrix;
  };

  return {gen_dense_matrix(50, 40), gen_dense_matrix(40, 30), "RandomMedium"};
}
}  // namespace

using TestType = TestCase;

class SparseCCSFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param.name;
  }

 protected:
  void SetUp() override {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  InType GetTestInputData() override {
    InType input;
    input.push_back(FromDense(test_case_.A));
    input.push_back(FromDense(test_case_.B));
    return input;
  }

  bool CheckTestOutputData(OutType &output) override {
    if (output.rows <= 0 || output.cols <= 0) {
      return false;
    }
    if (output.col_ptr.size() != static_cast<size_t>(output.cols + 1)) {
      return false;
    }
    if (output.values.size() != output.row_indices.size()) {
      return false;
    }
    if (output.col_ptr[0] != 0) {
      return false;
    }
    if (output.col_ptr.back() != static_cast<int>(output.values.size())) {
      return false;
    }

    for (size_t i = 1; i < output.col_ptr.size(); ++i) {
      if (output.col_ptr[i] < output.col_ptr[i - 1]) {
        return false;
      }
    }

    for (int row_idx : output.row_indices) {
      if (row_idx < 0 || row_idx >= output.rows) {
        return false;
      }
    }

    return true;
  }

 private:
  TestType test_case_;
};

namespace {
const TestCase kRandomMediumCase = GenerateRandomMedium();

const std::array<TestType, 2> kTestParams = {kSmallExact, kRandomMediumCase};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SparseMatrixMultiplicationCCSSeq, InType>(
                                               kTestParams, PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs),
                                           ppc::util::AddFuncTask<SparseMatrixMultiplicationCCSMPI, InType>(
                                               kTestParams, PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

TEST_P(SparseCCSFuncTest, HandlesCases) {
  ExecuteTest(GetParam());
}

const auto kPerfTestName = SparseCCSFuncTest::PrintFuncTestName<SparseCCSFuncTest>;

INSTANTIATE_TEST_SUITE_P(SparseCCSTests, SparseCCSFuncTest, kGtestValues, kPerfTestName);
}  // namespace
